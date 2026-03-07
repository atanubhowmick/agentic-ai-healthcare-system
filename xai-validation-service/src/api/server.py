from fastapi import APIRouter
from datamodel.models import DiagnosisValidationRequest, TreatmentValidationRequest, ValidationResponse, GenericResponse
from service.validator_service import run_diagnosis_validation, run_treatment_validation

router = APIRouter(prefix="/xai-validator")


@router.post("/validate-diagnosis", response_model=GenericResponse[ValidationResponse])
async def validate_diagnosis_endpoint(
    request: DiagnosisValidationRequest,
) -> GenericResponse[ValidationResponse]:
    return run_diagnosis_validation(request)


@router.post("/validate-treatment", response_model=GenericResponse[ValidationResponse])
async def validate_treatment_endpoint(
    request: TreatmentValidationRequest,
) -> GenericResponse[ValidationResponse]:
    return run_treatment_validation(request)
