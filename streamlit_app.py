import streamlit as st
import os
from main import ResearchAssistant, MainFunction
import asyncio
from core.types import ResearchDirection, EvaluationCriteria

def initialize_session_state():
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()

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
                st.warning("연구주제를 입력해주세요.")

if __name__ == "__main__":
    main()
