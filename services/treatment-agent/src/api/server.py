from fastapi import APIRouter
from datamodel.models import TreatmentRequest, TreatmentResponse, GenericResponse
from service.treatment_service import recommend
from log.logger import logger

router = APIRouter(prefix="/treatment-agent")


@router.post("/recommend", response_model=GenericResponse[TreatmentResponse])
async def recommend_treatment(request: TreatmentRequest) -> GenericResponse[TreatmentResponse]:
    logger.debug(
        "Received /recommend request | patient_id: %s | is_followup: %s",
        request.patient_id, request.is_followup,
    )
    treatment_response = recommend(request)
    logger.debug(
        "Returning treatment plan for patient %s | urgency: %s",
        request.patient_id, treatment_response.treatment.urgency,
    )
    return GenericResponse.success(treatment_response)
