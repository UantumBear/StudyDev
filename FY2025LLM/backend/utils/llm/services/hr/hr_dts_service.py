# backend/utils/llm/services/hr/hr_dts_service.py
from backend.utils.llm.services.llm_service_base import LLMServiceBase
from backend.utils.llm.prompts.hr.hr_dts_prompt import HrDtsPrompt

class HrDtsService(LLMServiceBase):
    """
        ※ 공식 챗봇이 아니며, 공부를 위해 만들어본 HR 컨셉의 챗봇입니다. 

        HRService 클래스는 LLMServiceBase를 상속한 HR 전용 RAG 챗봇 서비스 클래스이다.
        공통 LLM 호출 로직은 부모 클래스에서 제공
        HrDtsPrompt를 기본 persona로 설정하여 사내 규정, 인사 FAQ 등에 맞는 톤/규칙 적용
        send() 오버라이드는 필요 없음 (이모지 필터 불필요)        
    """
    def __init__(self, **kwargs):
        super().__init__(persona=HrDtsPrompt(), **kwargs)
