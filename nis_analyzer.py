import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import json
import time

# ==============================================================================
# [1] SYSTEM CONFIG & PROMPT ENGINEERING (뇌 설계)
# AI에게 우리의 '4단계 로직'과 '출력 형식'을 주입하는 시스템 프롬프트입니다.
# ==============================================================================

SYSTEM_PROMPT = """
당신은 '문학 시퀀스 분석 전문가'입니다. 사용자가 제공하는 [EBS 분석 이미지]와 [작품 줄거리/해석 텍스트]를 바탕으로 작품을 분석하여 JSON 데이터로 출력해야 합니다.

[분석 로직: 4단계 프로세스]
1단계: EBS 분석 이미지의 내용을 최우선 '기준 데이터'로 학습한다.
2단계: 제공된 전문 줄거리와 EBS 내용을 비교한다.
3단계: EBS에 있는 내용은 'EBS' 출처로, 없는 내용은 '줄거리(SUMMARY)' 출처로 분류한다.
4단계: 작품의 흐름을 논리적인 '시퀀스(장면)' 단위로 분할한다.

[출력 데이터 구조 - JSON 형식 엄수]
반드시 아래 JSON 스키마를 따르십시오. 다른 말은 하지 말고 JSON만 출력하세요.
{
    "project_name": "작품명",
    "sequences": [
        {
            "seq_id": "SEQ-01",
            "title": "장면 제목",
            "source_type": "EBS 또는 SUMMARY 중 하나 선택",
            "is_ebs_linked": true 또는 false,
            "macro_view": "이 시퀀스의 거시적 핵심 의미 (한 줄 요약)",
            "micro_detail": {
                "source_info": "[데이터 소스] 표시 (예: EBS 분석본 / 전문 개관)",
                "keywords": "분석 키워드 (예: 풍자, 해학, 골계미)",
                "scene_desc": "1. 장면 구성: 구체적인 상황 묘사",
                "deep_analysis": "2. 심층 해석: 이 장면의 의미와 상징성",
                "visual_point": "3. 시각화/연출 포인트: 분위기, 행동, 소품 등"
            }
        }
    ]
}
"""

# ==============================================================================
# [2] UI & FRONTEND (외관 설계)
# ==============================================================================
st.set_page_config(page_title="NIS 문학 분석 엔진", page_icon="🧬", layout="wide")

# 스타일 커스텀
st.markdown("""
<style>
    .report-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #2c3e50; margin-bottom: 20px; }
    .header-text { color: #2c3e50; font-weight: bold; }
    .ebs-badge { background-color: #3498db; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; }
    .summary-badge { background-color: #95a5a6; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ 엔진 컨트롤 패널")
    
    # API 키 입력 (유동적 분석을 위해 필수)
    api_key = st.text_input("Google API Key 입력", type="password", help="aistudio.google.com에서 무료 발급 가능")
    
    st.divider()
    
    with st.form("input_form"):
        st.subheader("1. 분석 대상")
        project_name = st.text_input("작품명", value="수궁가")
        
        st.subheader("2. 데이터 업로드")
        uploaded_images = st.file_uploader("EBS 분석 교재 (이미지)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        full_text = st.text_area("전문 줄거리 및 해석 (텍스트)", height=200, placeholder="여기에 작품의 전체 줄거리와 해석을 붙여넣으세요.")
        
        submit_btn = st.form_submit_button("🚀 분석 시작 (Run Analysis)", type="primary")

# ==============================================================================
# [3] MAIN LOGIC (분석 실행)
# ==============================================================================
st.title(f"🧬 NIS 문학 시퀀스 분석 엔진: {project_name}")
st.markdown("EBS 분석 자료와 전문 텍스트를 결합하여 **<시퀀스 마스터플랜>**을 생성합니다.")
st.divider()

if submit_btn:
    if not api_key:
        st.error("🚨 API Key가 필요합니다. 사이드바에 키를 입력해주세요.")
    elif not full_text:
        st.warning("⚠️ 분석할 텍스트(줄거리)를 입력해주세요.")
    else:
        try:
            # 1. Gemini 모델 설정
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash') # 이미지 처리에 강한 모델

            # 2. 프롬프트 구성
            input_content = [SYSTEM_PROMPT]
            input_content.append(f"작품명: {project_name}")
            input_content.append(f"전문 텍스트: {full_text}")
            
            # 이미지 추가
            if uploaded_images:
                st.info(f"📸 EBS 이미지 {len(uploaded_images)}장 로드 중...")
                for img_file in uploaded_images:
                    img = Image.open(img_file)
                    input_content.append(img)
            
            input_content.append("위 자료들을 종합하여 4단계 로직으로 분석한 결과를 JSON으로 출력하라.")

            # 3. AI 분석 요청
            with st.spinner('🧠 Gemini 엔진이 4단계 로직(EBS-줄거리 비교)을 수행 중입니다...'):
                response = model.generate_content(input_content)
                
                # JSON 파싱 (AI가 마크다운 코드블록을 쓸 경우 대비)
                result_text = response.text.replace("```json", "").replace("```", "").strip()
                result_json = json.loads(result_text)
                sequences = result_json.get("sequences", [])

            # 4. 결과 출력
            st.success(f"✅ 분석 완료! {len(sequences)}개의 시퀀스가 생성되었습니다.")
            
            # 탭 구성
            tab1, tab2 = st.tabs(["📝 정밀 분석 보고서 (Text)", "📊 시퀀스 구조표 (Table)"])
            
            # [TAB 1] 사용자가 요청한 텍스트 포맷 출력
            with tab1:
                st.subheader("4-1. 거시적 시퀀스 구조 (Macro View)")
                macro_text = ""
                for seq in sequences:
                    source_label = "EBS" if seq['source_type'] == "EBS" else "개관/요약"
                    macro_text += f"**{seq['seq_id']} [{seq['title']}]** (출처: {source_label})\n"
                    macro_text += f"- 핵심 의미: {seq['macro_view']}\n\n"
                st.info(macro_text)
                
                st.markdown("---")
                
                st.subheader("4-2. 미시적 시퀀스 정밀 분석 (Micro Detail)")
                for seq in sequences:
                    # 카드 UI
                    with st.expander(f"🎬 {seq['seq_id']}. {seq['title']}", expanded=True):
                        content = seq['micro_detail']
                        
                        # 요청하신 포맷대로 문자열 조합
                        formatted_text = f"""
**[데이터 소스]** {content['source_info']}
**[분석 키워드]** {content['keywords']}

**1. 장면 구성 (Scene):**
{content['scene_desc']}

**2. 심층 해석 (Deep Analysis):**
{content['deep_analysis']}

**3. 시각화/연출 포인트:**
{content['visual_point']}
                        """
                        st.markdown(formatted_text)

            # [TAB 2] 요청하신 시각화 표 (Table)
            with tab2:
                st.subheader("📑 시퀀스 데이터 매핑 테이블")
                
                table_data = []
                for seq in sequences:
                    table_data.append({
                        "시퀀스 ID": seq['seq_id'],
                        "장면명": seq['title'],
                        "데이터 소스": seq['source_type'], # EBS or SUMMARY
                        "EBS 연계 여부": "✅ 연계" if seq['is_ebs_linked'] else "⬜ 비연계",
                        "핵심 포인트": seq['macro_view']
                    })
                
                df = pd.DataFrame(table_data)
                
                # 데이터프레임 스타일링 (Source에 따라 색상 칩 적용)
                st.data_editor(
                    df,
                    column_config={
                        "데이터 소스": st.column_config.SelectboxColumn(
                            "데이터 소스",
                            help="분석의 근거가 된 자료",
                            width="medium",
                            options=["EBS", "SUMMARY"],
                            required=True,
                        ),
                        "핵심 포인트": st.column_config.TextColumn(
                            "핵심 포인트",
                            width="large"
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    disabled=True
                )
                
                # 통계 메트릭
                ebs_count = len([s for s in sequences if s['is_ebs_linked']])
                col1, col2 = st.columns(2)
                col1.metric("총 시퀀스 수", f"{len(sequences)}개")
                col2.metric("EBS 연계 구간", f"{ebs_count}개", delta="집중 분석 필요")

        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
            st.warning("팁: API 키가 정확한지, 인터넷 연결이 되어 있는지 확인하세요.")

else:
    # 초기 화면 안내
    st.info("👈 왼쪽 사이드바에 **API Key**, **EBS 이미지**, **작품 텍스트**를 넣고 실행하세요.")
    st.markdown("""
    #### 💡 사용 가이드
    1. **Google API Key**를 입력합니다. (없다면 무료 발급)
    2. **새로운 작품(예: 관동별곡)**의 EBS 분석 이미지를 업로드합니다.
    3. 인터넷에서 긁어온 **전문 줄거리/해석**을 텍스트 창에 붙여넣습니다.
    4. **[분석 시작]**을 누르면, Gemini가 4단계 로직으로 **자동 분석**하여 결과를 표와 리포트로 만들어줍니다.
    """)
    