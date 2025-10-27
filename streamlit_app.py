import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# ========================
# 앱 설정 & 전역
# ========================
st.set_page_config(page_title="미팅파티 자동 매칭 시스템", layout="wide")

WISH_COLS_ALL = ["1지망", "2지망", "3지망", "4지망"]
PRIORITY_FM = [
    ("1지망","1지망"), ("1지망","2지망"), ("2지망","1지망"),
    ("1지망","3지망"), ("3지망","1지망"), ("2지망","2지망"),
    ("2지망","3지망"), ("3지망","2지망"), ("3지망","3지망"),
    ("1지망","4지망"), ("4지망","1지망"), ("2지망","4지망"),
    ("4지망","2지망"), ("3지망","4지망"), ("4지망","3지망"), ("4지망","4지망"),
]
PRIO_INDEX = { f"{f}-{m}": i for i,(f,m) in enumerate(PRIORITY_FM, start=1) }
FEMALE_SEP_KEYWORDS = {k.lower() for k in ["여성참가자","여성","여자참가자","여자","woman","female"]}
COL_MAP = {
    '번호':'번호','no':'번호','id':'번호','NO':'번호','Id':'번호',
    '성명':'이름','이름':'이름','name':'이름','Name':'이름',
    '1지망':'1지망','희망1':'1지망','1순위':'1지망',
    '2지망':'2지망','희망2':'2지망','2순위':'2지망',
    '3지망':'3지망','희망3':'3지망','3순위':'3지망',
    '4지망':'4지망','희망4':'4지망','4순위':'4지망',
}

# 세션 상태
if "page" not in st.session_state: st.session_state.page = "업로드"
if "result" not in st.session_state: st.session_state.result = None

# ========================
# 스타일
# ========================
st.markdown(
    """
<style>
h1, h2, h3 { font-weight: 800; }
.block-container { padding-top: 1.5rem; }
.upload-box { border: 2px dashed #d0d7de; border-radius: 16px; padding: 80px 24px; text-align:center; color:#6b7280;}
.metric-card { background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:24px; }
.metric-title { color:#6b7280; font-size:14px; }
.metric-value { font-size:44px; font-weight:800; margin-top:6px; }
.progress-wrap { margin-top: 12px; }
.result-card { border:1px solid #e5e7eb; border-radius:14px; }
.stDataFrame { border: none; }
</style>
""",
    unsafe_allow_html=True,
)

# ========================
# 유틸/파싱
# ========================
def norm(v):
    if v is None: return ""
    return str(v).strip()

def is_sep_row(row_values):
    text = " ".join(norm(c) for c in row_values if c is not None).strip().lower()
    if not text: return False
    return any(k in text for k in FEMALE_SEP_KEYWORDS)

def is_header_row(row_values):
    cells = [norm(c).lower() for c in row_values]
    has_no = ('번호' in cells) or ('no' in cells) or ('id' in cells)
    has_name = ('성명' in cells) or ('이름' in cells) or ('name' in cells)
    return has_no and has_name

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        key = norm(c); low = key.lower()
        mapped = COL_MAP.get(key, COL_MAP.get(low, key))
        new_cols.append(mapped)
    df.columns = new_cols
    keep = ['번호','이름'] + [c for c in WISH_COLS_ALL if c in df.columns]
    return df[[c for c in keep if c in df.columns]]

def clean_wish(v, target_prefix):
    if v is None or (isinstance(v, float) and pd.isna(v)): return None
    s = str(v).strip()
    if s in ("", "—", "-", "--", "ㅡ", "–"): return None
    try:
        num = int(float(s))
        return f"{target_prefix}{num}"
    except Exception:
        return None

def extract_tables_from_sheet(ws_df: pd.DataFrame):
    df = ws_df.copy().fillna("")
    # 여성 구분 행 탐지
    sep_idx = None
    for i in range(len(df)):
        if is_sep_row(df.iloc[i].tolist()):
            sep_idx = i; break

    def find_header(df_part: pd.DataFrame):
        for i in range(len(df_part)):
            if is_header_row(df_part.iloc[i].tolist()):
                return i, [norm(c) for c in df_part.iloc[i].tolist()]
        return None, None

    men_df = pd.DataFrame(); women_df = pd.DataFrame()
    if sep_idx is not None:
        men_part = df.iloc[:sep_idx]; women_part = df.iloc[sep_idx+1:]
        m_top, m_cols = find_header(men_part); f_top, f_cols = find_header(women_part)
        if m_cols: men_df = pd.DataFrame(men_part.iloc[m_top+1:].values, columns=m_cols)
        if f_cols: women_df = pd.DataFrame(women_part.iloc[f_top+1:].values, columns=f_cols)
    else:
        top1, cols1 = find_header(df)
        if cols1:
            rest = df.iloc[top1+1:]; top2, cols2 = find_header(rest)
            if cols2:
                men_df = pd.DataFrame(df.iloc[top1+1: top1+1+top2].values, columns=cols1)
                women_df = pd.DataFrame(rest.iloc[top2+1:].values, columns=cols2)

    if not men_df.empty:
        men_df.dropna(how='all', inplace=True); men_df = standardize_columns(men_df)
    if not women_df.empty:
        women_df.dropna(how='all', inplace=True); women_df = standardize_columns(women_df)
    return men_df, women_df

def normalize_and_prefix(men_raw: pd.DataFrame, women_raw: pd.DataFrame):
    men = men_raw.copy(); women = women_raw.copy()
    men['번호'] = pd.to_numeric(men['번호'], errors='coerce').astype('Int64')
    women['번호'] = pd.to_numeric(women['번호'], errors='coerce').astype('Int64')
    men = men.dropna(subset=['번호']); women = women.dropna(subset=['번호'])
    men['번호'] = "M" + men['번호'].astype(int).astype(str)
    women['번호'] = "F" + women['번호'].astype(int).astype(str)

    men_ids = set(men['번호']); women_ids = set(women['번호'])
    for col in WISH_COLS_ALL:
        if col in men.columns:
            men[col] = men[col].apply(lambda v: clean_wish(v, "F"))
            men[col] = men[col].where(men[col].isin(women_ids), None)
        if col in women.columns:
            women[col] = women[col].apply(lambda v: clean_wish(v, "M"))
            women[col] = women[col].where(women[col].isin(men_ids), None)
    return men, women

# ========================
# 매칭/검증
# ========================
def build_candidates(men: pd.DataFrame, women: pd.DataFrame):
    candidates = []
    men_map = {r['번호']: r for _, r in men.iterrows()}

    for fcol, mcol in PRIORITY_FM:
        if (fcol not in women.columns) or (mcol not in men.columns):
            continue
        for _, w in women.iterrows():
            f_id = w['번호']
            m_id = w.get(fcol, None)
            if not (isinstance(m_id, str) and m_id.startswith('M')):
                continue
            m = men_map.get(m_id)
            if m is None:
                continue

            if m.get(mcol, None) == f_id:
                strong = (fcol == mcol)
                prio_id = PRIO_INDEX.get(f"{fcol}-{mcol}", 10**6)  # 안전장치
                # 길이 7 유지!
                candidates.append((
                    f"{fcol}-{mcol}",  # 0: rank_fm (예: "2지망-1지망")
                    m_id,              # 1
                    m['이름'],         # 2
                    f_id,              # 3
                    w['이름'],         # 4
                    strong,            # 5
                    prio_id            # 6  ← rank_weight 대신 prio_id(정수)
                ))
    return candidates

def dedupe_conflicts(candidates):
    def sort_key(p):
        prio_id = p[6]              # 마지막 요소: prio_id
        strong_first = 0 if p[5] else 1  # strong=True 우선 (동순위 tie-break)
        mnum = int(p[1][1:])        # 남성 번호 오름차순 보조
        return (prio_id, strong_first, mnum)

    kept = []
    discarded = []
    used_m = set()
    used_f = set()

    for p in sorted(candidates, key=sort_key):
        _, m_id, _, f_id, _, _, _ = p
        if (m_id in used_m) or (f_id in used_f):
            # (p, 사유) 형태를 유지 — 이후 build_output_tables에서 사유를 무시해도 됩니다
            discarded.append((p, "동일 참가자 중 낮은 우선순위"))
            continue
        kept.append(p)
        used_m.add(m_id)
        used_f.add(f_id)

    return kept, discarded

def cross_validate(kept, men, women):
    issues = []
    men_dic = {r['번호']: r for _, r in men.iterrows()}
    women_dic = {r['번호']: r for _, r in women.iterrows()}

    def wished(row, target_id):
        if row is None: return False
        for c in WISH_COLS_ALL:
            if c in row.index and row[c] == target_id: return True
        return False

    for p in kept:
        _, m_id, _, f_id, _, _, _ = p
        ok_m = wished(men_dic.get(m_id), f_id)
        ok_w = wished(women_dic.get(f_id), m_id)
        if not (ok_m and ok_w):
            reason_parts = []
            if not ok_m: reason_parts.append("남성 지망표에 여성 없음")
            if not ok_w: reason_parts.append("여성 지망표에 남성 없음")
            issues.append((m_id, f_id, " / ".join(reason_parts)))
    return issues

def build_output_tables(kept, discarded):
    # ----- 확정 결과(미리보기/엑셀)
    rows = []
    for p in kept:
        rank_fm, m_id, m_name, f_id, f_name, _, _ = p
        # 기존 rank_fm(F-M)을 M-F로 변환
        try:
            fcol, mcol = rank_fm.split('-', 1)
        except ValueError:
            fcol, mcol = rank_fm, ""
        rank_mf = f"{mcol}-{fcol}"

        rows.append({
            '순위(M-F)': rank_mf,
            '번호(M)': int(m_id[1:]),
            '이름(M)': m_name,
            '번호(F)': int(f_id[1:]),
            '이름(F)': f_name
        })

    out_df = pd.DataFrame(rows, columns=['순위(M-F)','번호(M)','이름(M)','번호(F)','이름(F)'])
    if not out_df.empty:
        out_df = out_df.sort_values('번호(M)').reset_index(drop=True)
        # 커플 순번 부여
        out_df.insert(0, '커플 순번', range(1, len(out_df)+1))

    # ----- 폐기 기록(남성 먼저 + 순위만 표시)
    disc_rows = []
    for item in discarded:
        p, _ = item
        rank_fm, m_id, m_name, f_id, f_name, _, _ = p
        try:
            fcol, mcol = rank_fm.split('-', 1)
        except ValueError:
            fcol, mcol = rank_fm, ""
        rank_mf = f"{mcol}-{fcol}"

        disc_rows.append({
            '순위(M-F)': rank_mf,
            '번호(M)': int(m_id[1:]),
            '이름(M)': m_name,
            '번호(F)': int(f_id[1:]),
            '이름(F)': f_name
        })

    discard_df = pd.DataFrame(
        disc_rows,
        columns=['순위(M-F)','번호(M)','이름(M)','번호(F)','이름(F)']
    )
    if not discard_df.empty:
        discard_df = discard_df.sort_values('번호(M)').reset_index(drop=True)

    return out_df, discard_df

def process_file(file) -> dict:
    ws_df = pd.read_excel(file, header=None)
    men_raw, women_raw = extract_tables_from_sheet(ws_df)
    if men_raw.empty or women_raw.empty or '번호' not in men_raw.columns or '번호' not in women_raw.columns or '이름' not in men_raw.columns or '이름' not in women_raw.columns:
        raise ValueError("파일을 읽을 수 없어 분석을 진행하지 못했습니다.")

    men_orig, women_orig = men_raw.copy(), women_raw.copy()
    men, women = normalize_and_prefix(men_raw, women_raw)

    candidates = build_candidates(men, women)
    kept, discarded = dedupe_conflicts(candidates)

    # 1회 자동 재검증 시도
    issues = cross_validate(kept, men, women)
    if issues:
        men, women = normalize_and_prefix(men_orig, women_orig)
        candidates = build_candidates(men, women)
        kept, discarded = dedupe_conflicts(candidates)
        issues = cross_validate(kept, men, women)
    if issues:
        msgs = [f"{m}\u2194{f} ({reason})" for m,f,reason in issues]
        raise ValueError("검증 오류 발생: " + " / ".join(msgs))

    preview_df, discard_df = build_output_tables(kept, discarded)
    N_m = len(men); N_f = len(women)
    total_people = N_m + N_f
    total_pairs = min(N_m, N_f)
    matched_pairs = len(preview_df)
    matched_people = matched_pairs * 2
    match_rate = round((matched_pairs/total_pairs*100) if total_pairs else 0, 1)

    cols_detected = (
        "1~4지망" if '4지망' in set(men.columns).union(set(women.columns)) else
        ("1~3지망" if '3지망' in set(men.columns).union(set(women.columns)) else
         ("1~2지망" if '2지망' in set(men.columns).union(set(women.columns)) else "1지망"))
    )
    run_header = f"Run ID({datetime.now().strftime('%Y-%m-%d %H:%M')} KST) | 파일명 {file.name} | 남({N_m}) 여({N_f}) | 컬럼 감지: {cols_detected}"

    # 엑셀 생성(폐기 기록 제외)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        preview_df.to_excel(writer, sheet_name="최종_커플_매칭", index=False)
        ws = writer.sheets["최종_커플_매칭"]
        base = len(preview_df) + 3
        ws.write(base, 0, f"총 {total_people}명({total_pairs}쌍) 중 {matched_people}명({matched_pairs}쌍) 썸매칭 성공.")
        ws.write(base+1, 0, f"{match_rate}%")
    excel_bytes = bio.getvalue()

    return {
        # 표/파일
        "preview_df": preview_df,
        "discard_df": discard_df,
        "excel_bytes": excel_bytes,
        "excel_filename": f"썸매칭_결과_{datetime.now().strftime('%Y%m%d')}.xlsx",
        # 통계
        "total_people": total_people,
        "total_pairs": total_pairs,
        "matched_pairs": matched_pairs,
        "matched_people": matched_people,
        "match_rate": match_rate,
        # 부가 정보
        "run_header": run_header,
    }

# ========================
# 화면 렌더링
# ========================
st.header("❤️ 미팅파티 자동 매칭 시스템")

if st.session_state.page == "업로드":
    # 업로더 영역을 박스 전체 클릭으로
    st.markdown(
        """
        <style>
        [data-testid="stFileUploader"] > section {
            border: 2px dashed #cbd5e1 !important;
            background: #f3f4f6 !important;      /* 회색 */
            border-radius: 16px !important;
            padding: 80px 24px !important;        /* 크게 */
            transition: border-color .2s ease, background .2s ease;
        }
        [data-testid="stFileUploader"] > section:hover {
            border-color: #94a3b8 !important;
            background: #eef2f7 !important;
            cursor: pointer !important;
        }
        [data-testid="stFileUploader"] label { display: none !important; }
        [data-testid="stFileUploader"] div[role="button"] { margin: 0 auto !important; }
        .uploader-title { text-align: center; color:#374151; font-size: 22px; font-weight: 800; margin-top: 8px; }
        .uploader-sub { text-align: center; color:#6b7280; font-size: 14px; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        label="엑셀 파일 업로드",
        type=["xlsx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="xlsx_uploader",
    )

    # 업로더 아래 안내 문구
    st.markdown('<div class="uploader-title">엑셀 파일을 드래그하거나 클릭하세요</div>', unsafe_allow_html=True)
    st.markdown('<div class="uploader-sub">.xlsx 파일만 지원됩니다</div>', unsafe_allow_html=True)

    # 📌 여기 추가
    st.markdown(
        """
        <div class="uploader-sub" style="margin-top: 8px;">
            <b>양식:</b><br>
            남성참가자<br>
            번호 / 이름 / 1지망 / 2지망 / 3지망<br>
            (빈 행)<br>
            여성참가자<br>
            번호 / 이름 / 1지망 / 2지망 / 3지망<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    if uploaded is not None:
        try:
            res = process_file(uploaded)
            st.session_state.result = res
            st.session_state.page = "결과"
            st.success("매칭 완료! 결과 화면으로 이동합니다.")
            st.rerun()  # 자동 전환
        except Exception as e:
            st.error(str(e))

# ========== 결과 페이지 ==========
if st.session_state.page == "결과":
    res = st.session_state.get("result", {}) or {}

    preview_df     = res.get("preview_df", pd.DataFrame(columns=["순위F-M","번호(M)","이름(M)","번호(F)","이름(F)"]))
    discard_df     = res.get("discard_df", pd.DataFrame(columns=["순위","여성","남성","사유"]))
    total_people   = int(res.get("total_people", 0))
    total_pairs    = int(res.get("total_pairs", 0))
    matched_pairs  = int(res.get("matched_pairs", len(preview_df)))
    matched_people = int(res.get("matched_people", matched_pairs * 2))
    match_rate     = float(res.get("match_rate", round((matched_pairs/total_pairs*100),1) if total_pairs else 0.0))
    excel_bytes    = res.get("excel_bytes", b"")
    excel_filename = res.get("excel_filename", "썸매칭_결과.xlsx")

    # 러닝 헤더(선택)
    run_header = res.get("run_header")
    if run_header:
        st.caption(run_header)

    st.markdown("### 📊 최종 결과")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div style=\"padding:20px;border-radius:14px;background:#F9FAFB;border:1px solid #E5E7EB;\">
              <div style=\"color:#6B7280;font-weight:700;\">총 참가자</div>
              <div style=\"font-size:42px;font-weight:900;margin-top:6px;color:#111827;\">{total_people}</div>
              <div style=\"color:#6B7280;\">{total_pairs}쌍</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style=\"padding:20px;border-radius:14px;background:#F9FAFB;border:1px solid #E5E7EB;\">
              <div style=\"color:#6B7280;font-weight:700;\">매칭 성공</div>
              <div style=\"font-size:42px;font-weight:900;margin-top:6px;color:#111827;\">{matched_people}명</div>
              <div style=\"color:#6B7280;\">{matched_pairs}쌍</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
         f"""
        <div style="padding:20px;border-radius:14px;background:#F9FAFB;border:1px solid #E5E7EB;">
            <div style="color:#6B7280;font-weight:700;">매칭률</div>
            <div style="font-size:42px;font-weight:900;margin-top:6px;color:#111827;">{match_rate}%</div>
            <div style="height:8px;background:#F3F4F6;border-radius:999px;margin-top:8px;position:relative;">
                <div style="height:8px;background:#9CA3AF;border-radius:999px;width:{match_rate}%;"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### 👩‍❤️‍👨 최종 커플 매칭")

    # 다운로드 버튼을 미리보기 표 '위에' 배치
    st.download_button(
        label="📥 결과 엑셀 다운로드",
        data=excel_bytes,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    # 폐기 기록 - 기본 접힘 + 우측 화살표
    st.markdown(
        """
        <style>
        details.st-expander > summary { list-style: none; position: relative; padding-right: 24px !important; }
        details.st-expander > summary::marker { content: ""; }
        details.st-expander > summary:after {
            content: "⌄"; position: absolute; right: 0; top: 50%; transform: translateY(-50%) rotate(0deg);
            transition: transform .2s ease; color:#6B7280; font-size: 18px;
        }
        details.st-expander[open] > summary:after { transform: translateY(-50%) rotate(180deg); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("폐기 기록", expanded=False):
        if discard_df is None or len(discard_df) == 0:
            st.caption("(해당 없음)")
        else:
            st.dataframe(discard_df, use_container_width=True, hide_index=True)

    # =======================
    # 업로드 화면으로 돌아가기 버튼 (결과 화면에서만)
    # =======================
    st.markdown("---")
    if st.button("⬅️ 업로드로 돌아가기", use_container_width=True):
        st.session_state.page = "업로드"
        st.session_state.result = None
        st.rerun()
        


