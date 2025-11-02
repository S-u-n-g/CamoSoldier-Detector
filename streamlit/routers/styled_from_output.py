# styled_from_llm.py
import os, re, datetime as dt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem, Table, TableStyle, Image, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle
from reportlab.graphics.shapes import Drawing, Rect

def read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# =========================
# 0) 폰트/경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 fonts/ 폴더에 TTF가 있으면 우선 사용, 없으면 Windows 맑은고딕 시도
CANDIDATE_FONTS = [
    os.path.join(BASE_DIR, "fonts", "malgun.ttf"),
    r"C:\Windows\Fonts\malgun.ttf"  # Windows
]
FONT_PATH = next((p for p in CANDIDATE_FONTS if os.path.exists(p)), None)
if not FONT_PATH:
    raise FileNotFoundError(
        "한글 폰트를 찾을 수 없습니다. fonts/NanumGothic.ttf를 추가하거나 "
        "Windows라면 C:\\Windows\\Fonts\\malgun.ttf 경로를 확인하세요."
    )
pdfmetrics.registerFont(TTFont("KR", FONT_PATH))

# =========================
# 1) 스타일 정의
# =========================


def build_styles():
    styles = getSampleStyleSheet()
    # 기본 폰트를 모두 KR로 교체
    for k in styles.byName:
        styles[k].fontName = "KR"
    styles.add(ParagraphStyle("TitleKR", parent=styles["Title"], fontName="KR",
                              fontSize=22, leading=28, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle("H1KR", parent=styles["Heading1"], fontName="KR",
                              fontSize=16, leading=22, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle("H2KR", parent=styles["Heading2"], fontName="KR",
                              fontSize=13, leading=18, spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle("BodyKR", parent=styles["Normal"], fontName="KR",
                              fontSize=10.5, leading=16, spaceAfter=4))
    styles.add(ParagraphStyle("Muted", parent=styles["Normal"], fontName="KR",
                              fontSize=9, leading=13, textColor=colors.HexColor("#666")))
    styles.add(ParagraphStyle("KPI", parent=styles["Normal"], fontName="KR",
                              fontSize=12, leading=16, alignment=TA_CENTER))
    return styles


# =========================
# 2) 간단 MD 처리 유틸
# =========================
BULLET_PAT = re.compile(r'^\s*[-*]\s+')
H1_PAT = re.compile(r'^\s*#\s+(.+?)\s*$')
H2_PAT = re.compile(r'^\s*##\s+(.+?)\s*$')

# =========================
# 2.5) 대표 사례 파싱 + 카드 렌더러
# =========================
CASE_LINE_RE = re.compile(
    r"""^\s*-\s*`(?P<id>[^`]+)`,\s*
        (?P<utc>[^()]+?)\s*
        \(\s*KST\s*:\s*(?P<kst>[^)]+)\)\s*,\s*
        score\s*:\s*(?P<score>[0-9.]+)\s*,\s*
        caption\s*:\s*["“](?P<caption>.*?)["”]\s*$
    """,
    re.VERBOSE,
)

KST = dt.timezone(dt.timedelta(hours=9))

# =============== MD 표 파싱 + 테이블 렌더 ===============

MD_TABLE_SEP = re.compile(r'^\s*\|\s*-')

HOUR_RE = re.compile(r'^(\d{1,2})\s*시$')

def is_time_distribution_table(rows: list[list[str]]) -> bool:
    """첫 컬럼이 '00시' 형식으로 이어지는지 대략 판별."""
    if not rows or len(rows[0]) < 2:
        return False
    data = rows[1:]
    hit = 0
    for r in data:
        if r and HOUR_RE.match(r[0]):
            hit += 1
    return hit >= max(4, len(data)//2)  # 데이터 절반 이상이 '시'면 시간표로 간주

def split_hours_rows(rows: list[list[str]]):
    """rows(헤더 포함)를 0–11시 / 12–23시 두 묶음으로 분할해서 (left_rows, right_rows, all_data_rows) 반환."""
    header = rows[0]
    data = rows[1:]
    parsed = []
    for r in data:
        if not r:
            continue
        m = HOUR_RE.match(r[0])
        if not m:
            continue
        h = int(m.group(1))
        parsed.append((h, r))
    parsed.sort(key=lambda x: x[0])

    left_data = [r for h, r in parsed if 0 <= h <= 11]
    right_data = [r for h, r in parsed if 12 <= h <= 23]
    left_rows = [header] + left_data if left_data else []
    right_rows = [header] + right_data if right_data else []
    all_data_rows = [r for _, r in parsed]
    return left_rows, right_rows, all_data_rows

def parse_md_table(lines, start_idx):
    """
    lines[start_idx]가 '|'로 시작하는 표 헤더라면,
    마크다운 표 블록을 파싱해 (rows(list[list[str]]), next_index)를 반환.
    rows[0]은 헤더. 구분선(| --- | --- |)은 자동 건너뜀.
    """
    rows = []
    i = start_idx
    while i < len(lines):
        ln = lines[i]
        if not ln.strip().startswith("|"):
            break
        # 파이프 기준 split하되 양끝 파이프 제거
        parts = [c.strip() for c in ln.strip().strip("|").split("|")]
        # 구분선은 패스
        if MD_TABLE_SEP.match(ln):
            i += 1
            continue
        rows.append(parts)
        i += 1
    return rows, i

def _to_int_safe(cell: str) -> int:
    # "2건" → 2, "0" → 0 같은 케이스 처리
    try:
        return int(re.findall(r'\d+', cell)[0])
    except Exception:
        return 0

def _mini_bar(width, height, value, vmax):
    """
    셀 안에 넣을 미니 바차트용 Drawing 반환.
    width/height는 포인트(px) 단위. value/vmax로 폭 비례.
    """
    d = Drawing(width, height)
    if vmax > 0 and value > 0:
        w = max(1, width * (value / vmax))
        d.add(Rect(0, 0, w, height, fillColor=colors.HexColor("#a3c4f3"), strokeWidth=0))
    # 배경 라인(옅은 테두리)
    d.add(Rect(0, 0, width, height, fillColor=None, strokeColor=colors.HexColor("#d9d9d9"), strokeWidth=0.5))
    return d

def build_table_flowable(rows, styles, add_bar=False, bar_col_width=70, row_height=16, bar_vmax=None):
    """
    rows: 2D list (rows[0] = header)
    add_bar=True면 마지막에 '시각화' 컬럼을 추가하고 미니 바 넣음(두 번째 컬럼 숫자 기준).
    """
    if not rows:
        return None

    # 헤더/데이터 분리
    header = rows[0]
    data = rows[1:] if len(rows) > 1 else []

    # 숫자 컬럼(여기서는 2번째 컬럼)을 찾아 막대용 수치/최대값 계산
    counts = []
    if add_bar and len(header) >= 2:
        for r in data:
            counts.append(_to_int_safe(r[1] if len(r) > 1 else "0"))
        vmax = bar_vmax if (bar_vmax is not None) else (max(counts) if counts else 0)
        # '시각화' 컬럼 추가
        header = header + ["시각화"]
        new_data = []
        for idx, r in enumerate(data):
            bar = _mini_bar(bar_col_width, row_height-4, counts[idx], vmax)
            new_data.append(r + [bar])
        data = new_data

    table_data = [header] + data

    # 폭 계산: 텍스트 컬럼은 자동, 바 컬럼은 고정폭
    col_widths = [None] * len(header)
    if add_bar:
        col_widths[-1] = bar_col_width  # 마지막 컬럼 고정폭

    tbl = Table(table_data, colWidths=col_widths, hAlign="LEFT")
    # 스타일
    tbl.setStyle(TableStyle([
        ('FONT', (0,0), (-1,-1), 'KR', 10),
        ('LEADING', (0,0), (-1,-1), 13),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),  # 헤더 가운데
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEABOVE', (0,0), (-1,0), 0.6, colors.HexColor("#333333")),
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.HexColor("#333333")),
        ('LINEBELOW', (0,-1), (-1,-1), 0.6, colors.HexColor("#cccccc")),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        # 지브라 스트라이프
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('BACKGROUND', (0,1), (-1,-1), colors.Color(0,0,0,0.0)),
    ]))
    # 지브라를 번갈아 칠하기
    for i in range(1, len(table_data)):
        if i % 2 == 1:
            tbl.setStyle(TableStyle([('BACKGROUND', (0,i), (-1,i), colors.HexColor("#f7f7f7"))]))

    # 숫자 정렬(두 번째 컬럼)
    if len(header) >= 2:
        tbl.setStyle(TableStyle([('ALIGN', (1,1), (1,-1), 'RIGHT')]))

    return tbl


def _to_kst_str(iso_s: str) -> str:
    d = dt.datetime.fromisoformat(iso_s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")

ID_LINE_RE = re.compile(r"^\s*-\s*`(?P<id>[^`]+)`\s*$")

def extract_case_ids(report_text: str) -> list[str]:
    ids, in_sec = [], False
    for ln in report_text.splitlines():
        if not in_sec:
            if re.match(r"^\s*#\s*대표\s*사례\s*$", ln):
                in_sec = True
            continue
        if re.match(r"^\s*#\s+", ln):
            break
        m = ID_LINE_RE.match(ln)
        if m:
            ids.append(m.group("id").strip())
    return ids

def build_case_cards_from_records(
    case_ids: list[str],
    records: list[dict],
    styles,
    result_dir: str = "./result",
    image_col_width_mm: float = 80.0,
    meta_col_width_mm: float = 90.0,
    max_image_height_mm: float = 60.0,
    gap_after_mm: float = 6.0,
):
    idx = {str(r.get("id")): r for r in records}
    body = styles["BodyKR"]
    iw, mw, mh = image_col_width_mm*mm, meta_col_width_mm*mm, max_image_height_mm*mm
    flows = []
    for cid in case_ids:
        r = idx.get(cid)
        if not r:
            # 레코드가 없으면 스킵(원하면 ‘누락’ 카드로 대체 가능)
            continue
        utc = str(r.get("timestamp", ""))
        kst = _to_kst_str(utc) if utc else ""
        score = r.get("score", None)
        score_s = f"{float(score):.2f}" if score is not None else "-"
        caption = r.get("caption", "")

        img_path = os.path.join(result_dir, f"{cid}.jpg")
        if os.path.exists(img_path):
            img = Image(img_path)
            img._restrictSize(iw, mh)
            img_cell = img
        else:
            img_cell = Paragraph(f"<b>{cid}.jpg</b><br/>이미지 없음", body)

        meta_html = (
            f"<b>{cid}</b><br/>"
            f"<b>UTC</b>: {utc}<br/>"
            f"<b>KST</b>: {kst}<br/>"
            f"<b>score</b>: {score_s}<br/>"
            f"<b>caption</b>: {caption}"
        )
        tbl = Table([[img_cell, Paragraph(meta_html, body)]], colWidths=[iw, mw], hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('BOX',(0,0),(-1,-1),0.25,colors.grey),
            ('INNERGRID',(0,0),(-1,-1),0.25,colors.whitesmoke),
        ]))
        flows.append(KeepTogether([tbl, Spacer(1, gap_after_mm*mm)]))
    return flows


def extract_representative_cases(report_text: str):
    """마크다운의 '# 대표 사례' 섹션에서 불릿 라인을 파싱해 dict 리스트로 반환."""
    lines = report_text.splitlines()
    cases = []
    in_section = False

    for ln in lines:
        if not in_section:
            if re.match(r"^\s*#\s*대표\s*사례\s*$", ln):
                in_section = True
            continue

        if re.match(r"^\s*#\s+", ln):  # 다음 섹션(# ...)이면 중단
            break

        m = CASE_LINE_RE.match(ln)
        print(ln)
        if m:
            d = {k: v.strip() for k, v in m.groupdict().items()}
            cases.append(d)

    return cases

def build_case_card_flowables(
    cases,
    styles,
    result_dir="./result",
    image_col_width_mm=80.0,
    meta_col_width_mm=90.0,
    max_image_height_mm=60.0,
    gap_after_mm=6.0,
):
    """각 사례를 (이미지 | 메타) 2열 테이블 카드로 만들어 Flowable 리스트 반환."""
    body = styles["BodyKR"]
    header = styles["H2KR"]

    flowables = []
    iw = image_col_width_mm * mm
    mw = meta_col_width_mm * mm
    mh = max_image_height_mm * mm

    for c in cases:
        cid = c.get("id", "")
        utc = c.get("utc", "")
        kst = c.get("kst", "")
        score = c.get("score", "")
        caption = c.get("caption", "")

        # 이미지 셀
        img_path = os.path.join(result_dir, f"{cid}.jpg")
        if os.path.exists(img_path):
            img = Image(img_path)
            img._restrictSize(iw, mh)  # 비율 유지 축소
            img_cell = img
        else:
            img_cell = Paragraph(f"<b>{cid}.jpg</b><br/>이미지를 찾을 수 없습니다.", body)

        # 메타 셀
        meta_html = (
            f"<b>{cid}</b><br/>"
            f"<b>UTC</b>: {utc}<br/>"
            f"<b>KST</b>: {kst}<br/>"
            f"<b>score</b>: {score}<br/>"
            f"<b>caption</b>: {caption}"
        )
        meta_para = Paragraph(meta_html, body)

        tbl = Table([[img_cell, meta_para]], colWidths=[iw, mw], hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('BOX', (0,0), (-1,-1), 0.25, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.whitesmoke),
        ]))

        flowables.append(KeepTogether([tbl, Spacer(1, gap_after_mm * mm)]))

    return flowables


def md_inline_to_htmlish(text: str) -> str:
    """**bold** / `code` 정도만 간단히 처리 (ReportLab Paragraph는 HTML-like 지원)."""
    # 굵게
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # 인라인 코드 → monospace 흉내
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    return text


def split_sections(report_text: str):
    """# 제목 기반으로 섹션을 분리. 반환: [(제목, 본문텍스트), ...]"""
    lines = report_text.strip().splitlines()
    sections = []
    cur_title = None
    cur_buf = []
    for ln in lines:
        m1 = H1_PAT.match(ln)
        m2 = H2_PAT.match(ln)
        if m1 and (cur_title is None or cur_buf):  # 새 # 헤딩 → 이전 섹션 저장
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_buf).strip()))
                cur_buf = []
            cur_title = m1.group(1).strip()
        elif m1:
            cur_title = m1.group(1).strip()
        elif m2:
            # ## 헤딩은 본문 내부 소제목 → 본문에 그대로 둠
            cur_buf.append(f"<H2>{m2.group(1).strip()}</H2>")
        else:
            cur_buf.append(ln)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_buf).strip()))
    return sections


def section_to_flowables(title: str, body: str, styles):
    """섹션 본문을 문단/불릿으로 변환."""
    flows = [Paragraph(title, styles["H1KR"])]
    # 소제목(<H2> 태그로 임시 표기) 분리
    chunks = re.split(r'(<H2>.*?</H2>)', body)
    for ch in chunks:
        if not ch:
            continue
        if ch.startswith("<H2>"):
            subt = ch.replace("<H2>", "").replace("</H2>", "")
            flows.append(Paragraph(subt, styles["H2KR"]))
            continue

        # 블록을 줄 단위로 읽어서 불릿/문단 구분
        lines = ch.splitlines()
        buf = []
        bullets = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            # 3-1) 마크다운 표 감지: '|'로 시작하고 다음 줄에 구분선이 오는지 확인
            if ln.strip().startswith("|") and (i + 1 < len(lines)) and MD_TABLE_SEP.match(lines[i + 1]):
                rows, next_i = parse_md_table(lines, i)
                # 어떤 표인지에 따라 시각화 컬럼(미니 바) 추가 여부 결정
                # 제목 문맥에 "시간 분포", "요일 분포"가 있으면 bar 추가
                title_lower = title.lower()
                # 섹션/소제목 문맥도 참고하도록 ch 이전 H2를 고려할 수도 있음(간단히 title만 사용)
                add_bar = ("시간 분포" in title) or ("요일 분포" in title)

                # ▶ 시간 분포 표이면 0–11 / 12–23 두 표로 분할하여 좌우 배치
                if is_time_distribution_table(rows):
                    left_rows, right_rows, all_data_rows = split_hours_rows(rows)

                    # 공통 vmax (좌우 막대 스케일 통일)
                    all_counts = [_to_int_safe(r[1] if len(r) > 1 else "0") for r in all_data_rows]
                    common_vmax = max(all_counts) if all_counts else 0

                    left_tbl = build_table_flowable(left_rows, styles, add_bar=True, bar_col_width=70,
                                                    bar_vmax=common_vmax)
                    right_tbl = build_table_flowable(right_rows, styles, add_bar=True, bar_col_width=70,
                                                     bar_vmax=common_vmax)

                    if bullets:
                        flows.append(ListFlowable(
                            [ListItem(Paragraph(md_inline_to_htmlish(b), styles["BodyKR"]), bulletText="•") for b in
                             bullets],
                            bulletType='bullet', leftIndent=12
                        ))
                        bullets = []
                    if buf:
                        flows.append(Paragraph(md_inline_to_htmlish("\n".join(buf)), styles["BodyKR"]))
                        buf = []

                    # 좌우 나란히 배치(외부 테이블)
                    outer = Table([[left_tbl, right_tbl]], colWidths=[None, None], hAlign="LEFT")
                    outer.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 12),  # 표 사이 여백
                        ('TOPPADDING', (0, 0), (-1, -1), 0),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                    ]))
                    flows.append(outer)
                    flows.append(Spacer(1, 6))
                    i = next_i
                    continue

                tbl = build_table_flowable(rows, styles, add_bar=add_bar)
                if tbl:
                    # 버퍼/불릿 덤프 후 표 삽입
                    if bullets:
                        flows.append(ListFlowable(
                            [ListItem(Paragraph(md_inline_to_htmlish(b), styles["BodyKR"]), bulletText="•") for b in
                             bullets],
                            bulletType='bullet', leftIndent=12
                        ))
                        bullets = []
                    if buf:
                        flows.append(Paragraph(md_inline_to_htmlish("\n".join(buf)), styles["BodyKR"]))
                        buf = []
                    flows.append(tbl)
                    flows.append(Spacer(1, 6))
                i = next_i
                continue

            # 3-2) 기존 불릿/문단 처리
            if not ln.strip():
                if bullets:
                    flows.append(ListFlowable(
                        [ListItem(Paragraph(md_inline_to_htmlish(b), styles["BodyKR"]), bulletText="•") for b in
                         bullets],
                        bulletType='bullet', leftIndent=12
                    ))
                    bullets = []
                if buf:
                    flows.append(Paragraph(md_inline_to_htmlish("\n".join(buf)), styles["BodyKR"]))
                    buf = []
                flows.append(Spacer(1, 2))
                i += 1
                continue

            if BULLET_PAT.match(ln):
                bullets.append(BULLET_PAT.sub("", ln).strip())
            else:
                buf.append(ln)
            i += 1

        # 남은 거 마무리
        if bullets:
            flows.append(ListFlowable(
                [ListItem(Paragraph(md_inline_to_htmlish(b), styles["BodyKR"]), bulletText="•") for b in bullets],
                bulletType='bullet', leftIndent=12
            ))
        if buf:
            flows.append(Paragraph(md_inline_to_htmlish("\n".join(buf)), styles["BodyKR"]))

    flows.append(Spacer(1, 8))
    return flows


# =========================
# 3) 헤더/푸터
# =========================
def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # 헤더 라인
    canvas.setStrokeColor(colors.HexColor("#d9d9d9"))
    canvas.line(16*mm, h-20*mm, w-16*mm, h-20*mm)
    canvas.setFont("KR", 10)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(16*mm, h-16.5*mm, doc._header_title)
    # 푸터 라인/페이지 번호
    canvas.setStrokeColor(colors.HexColor("#d9d9d9"))
    canvas.line(16*mm, 16*mm, w-16*mm, 16*mm)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(w-16*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


# =========================
# 4) 핵심: LLM 응답 텍스트 → 꾸민 PDF
# =========================
def llm_report_text_to_pdf(report_text: str, out_path: str = "styled_from_llm.pdf",
                           title: str = "위장군인 자동 보고서(LLM 응답 기반)", records: list[dict] | None = None):
    styles = build_styles()
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm, topMargin=22*mm, bottomMargin=18*mm
    )
    doc._header_title = title

    story = []

    # 표지
    story.append(Spacer(1, 30))
    # 첫 줄이 "# 제목" 이면 그걸 표지 제목으로 사용
    first_line = report_text.strip().splitlines()[0] if report_text.strip() else ""

    cover_title = title
    story.append(Paragraph(cover_title, styles["TitleKR"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(dt.datetime.now().strftime("생성 시각: %Y-%m-%d %H:%M (KST)"), styles["Muted"]))
    story.append(PageBreak())

    # 섹션 분해 및 렌더
    sections = split_sections(report_text)
    # 섹션 이름 정렬(원하는 순서가 있을 때)
    order = ["기간 요약", "핵심 지표", "패턴 분석", "대표 사례", "위험", "위험/권고", "주의", "주의/한계"]
    # 섹션을 이름 기반으로 우선 정렬, 나머지는 뒤에
    def sort_key(x):
        name = x[0]
        idx = min([order.index(o) for o in order if o in name] + [999])
        return (idx, name)
    sections.sort(key=sort_key)

    for title_i, body_i in sections:
        # story += section_to_flowables(title_i, body_i, styles)
        if re.search(r"대표\s*사례", title_i):
            story.append(Paragraph(title_i, styles["H1KR"]))
            story.append(Spacer(1, 4 * mm))
            # cases = extract_representative_cases(report_text)
            ids = extract_case_ids(report_text)
            if ids and records:
                story.extend(build_case_cards_from_records(ids, records, styles, "./result"))
            else:
                # records 미전달/파싱실패 시, 기존 텍스트로라도 출력
                story += section_to_flowables(title_i, body_i, styles)
        else:
            story += section_to_flowables(title_i, body_i, styles)
        #     print("cases:", cases)
        #     if cases:
        #         story.extend(build_case_card_flowables(
        #             cases,
        #             styles = styles,
        #             result_dir = "./result",
        #             image_col_width_mm = 80.0,
        #             meta_col_width_mm = 90.0,
        #             max_image_height_mm = 60.0,
        #             gap_after_mm = 6.0,
        #         ))
        #     else:
        #         # 파싱 실패/사례 없음 → 기존 본문을 그대로 출력(안전망)
        #         story += section_to_flowables(title_i, body_i, styles)
        # else:
        #     story += section_to_flowables(title_i, body_i, styles)

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF 생성 완료: {out_path}")


# =========================
# 5) 사용 예시 (직접 실행 시)
# =========================
if __name__ == "__main__":
    # 여기 report_text 자리에 LLM(resp.text) 문자열을 넣어 실행하세요.
    import sys

    default_txt = "C:\\Users\Home\PycharmProjects\\faseapi_for_project\압축\out\\report_20251101_142058.txt"
    txt_path = sys.argv[1] if len(sys.argv) > 1 else default_txt
    print(default_txt)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"LLM 리포트 텍스트 파일을 찾을 수 없습니다: {txt_path}\n"
            f"gemini_API.py에서 save_text_to_txt(report_text, 'report.txt')로 생성했는지 확인하세요."
        )

    # report.txt 읽기
    report_text = read_text_file(txt_path)

    # 첫 줄이 '# 제목'이면 그걸 PDF 표지로 사용하고, 아니면 기본 타이틀 사용
    first_line = report_text.strip().splitlines()[0] if report_text.strip() else ""
    title = first_line.lstrip("#").strip() if first_line.startswith("#") else "위장군인 자동 보고서(LLM 응답 기반)"

    # 출력 파일명: 입력 txt 이름을 기반으로 변경 (예: report.txt -> report_styled.pdf)
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    out_pdf = os.path.join(BASE_DIR, f"{base_name}_styled2.pdf")

    # PDF 생성
    llm_report_text_to_pdf(report_text, out_path=out_pdf, title=title)
