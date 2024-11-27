import streamlit as st
import os
from main import ResearchAssistant, MainFunction
import asyncio
from config.settings import LLM_CONFIG, EMBEDDING_CONFIG
from core.types import ResearchDirection, EvaluationCriteria

def initialize_session_state():
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
    if 'api_keys_set' not in st.session_state:
        st.session_state.api_keys_set = False

def save_api_keys(llm_key: str, voyage_key: str):
    """API 키를 설정에 저장"""
    LLM_CONFIG["api_key"] = llm_key
    EMBEDDING_CONFIG["api_key"] = voyage_key
    st.session_state.api_keys_set = True

def api_keys_sidebar():
    """API 키 입력을 위한 사이드바"""
    with st.sidebar:
        st.header("API 키 설정")

        if st.session_state.api_keys_set:
            st.success("API 키가 설정되었습니다.")

        with st.form("api_keys_form"):
            llm_key = st.text_input(
                "GROQ API 키",
                type="password",
                value=LLM_CONFIG["api_key"] if LLM_CONFIG["api_key"] != "place-holder" else ""
            )
            voyage_key = st.text_input(
                "VoyageAI API 키",
                type="password",
                value=EMBEDDING_CONFIG["api_key"] if EMBEDDING_CONFIG["api_key"] != "place-holder" else ""
            )

            if st.form_submit_button("API 키 저장"):
                if llm_key and voyage_key:
                    save_api_keys(llm_key, voyage_key)
                    st.success("API 키가 성공적으로 저장되었습니다.")
                else:
                    st.error("모든 API 키를 입력해주세요.")

def create_evaluation_criteria():
    """평가 기준 생성"""
    with st.expander("평가 기준 설정"):
        methodology_weight = st.slider("연구방법론 가중치", 0.0, 1.0, 0.3)
        innovation_weight = st.slider("혁신성 가중치", 0.0, 1.0, 0.2)
        validity_weight = st.slider("타당성 가중치", 0.0, 1.0, 0.3)
        reliability_weight = st.slider("신뢰성 가중치", 0.0, 1.0, 0.2)

        min_quality_score = st.slider("최소 품질 점수", 0.0, 1.0, 0.7)
        required_validity_score = st.slider("필요 타당성 점수", 0.0, 1.0, 0.8)

        return EvaluationCriteria(
            methodology_weight=methodology_weight,
            innovation_weight=innovation_weight,
            validity_weight=validity_weight,
            reliability_weight=reliability_weight,
            min_quality_score=min_quality_score,
            required_validity_score=required_validity_score
        )

def main():
    st.title('SUHANGSSALMUKDEMO')
    st.header("Welcome to Research Assistant CLI!")
    st.write("This tool will help you conduct research and analysis.")

    initialize_session_state()
    api_keys_sidebar()

    if not st.session_state.api_keys_set:
        st.warning("시작하기 전에 사이드바에서 API 키를 설정해주세요.")
        return

    with st.form(key='research_form'):
        query = st.text_input(label='연구주제')
        context = st.text_area(label='연구설명')

        # 연구 방향성 선택
        direction = st.selectbox(
            "연구 방향성",
            options=[d.value for d in ResearchDirection],
            format_func=lambda x: {
                'exploratory': '탐색적 연구',
                'descriptive': '기술적 연구',
                'explanatory': '설명적 연구',
                'experimental': '실험적 연구',
                'theoretical': '이론적 연구'
            }[x]
        )

        # 평가 기준 설정
        evaluation_criteria = create_evaluation_criteria()

        submit_button = st.form_submit_button(label='제출')

        if submit_button:
            if query:
                if st.session_state.api_keys_set:
                    result = MainFunction.process_and_display_results(
                        assistant=st.session_state.assistant,
                        topic=query,  # 연구 주제
                        description=context,  # 연구 설명
                        direction=ResearchDirection(direction),  # 연구 방향성
                        evaluation_criteria=evaluation_criteria  # 평가 기준
                    )
                    if result:
                        st.session_state.last_result = result
                else:
                    st.error("API 키를 먼저 설정해주세요.")
            else:
                st.warning("연구주제를 입력해주세요.")

if __name__ == "__main__":
    main()
