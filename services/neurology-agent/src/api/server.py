from fastapi import APIRouter
from datamodel.models import NeurologyRequest, NeurologyResponse, GenericResponse
from service.neurology_service import diagnose
from log.logger import logger

router = APIRouter(prefix="/neurology-agent")


@router.post("/diagnose", response_model=GenericResponse[NeurologyResponse])
async def diagnose_neurological_condition(request: NeurologyRequest) -> GenericResponse[NeurologyResponse]:
    logger.debug("Received /diagnose request | patient_id: %s | is_followup: %s",
                 request.patient_id, request.is_followup)
    diagnosis_response = diagnose(request)
    logger.debug("Returning diagnosis for patient %s", request.patient_id)
    return GenericResponse.success(diagnosis_response)
