from fastapi import APIRouter
from datamodel.models import DiagnosisRequest, DiagnosisResponse, GenericResponse
from service.cardiology_service import diagnose
from log.logger import logger

router = APIRouter(prefix="/cardiology-agent")


@router.post("/diagnose", response_model=GenericResponse[DiagnosisResponse])
async def diagnose_heart_condition(request: DiagnosisRequest) -> GenericResponse[DiagnosisResponse]:
    logger.debug("Received /diagnose request | patient_id: %s | is_followup: %s",
                 request.patient_id, request.is_followup)
    diagnosis_response = diagnose(request)
    logger.debug("Returning diagnosis for patient %s", request.patient_id)
    return GenericResponse.success(diagnosis_response)
