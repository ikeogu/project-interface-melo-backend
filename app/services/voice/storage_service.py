"""
Storage service — S3/R2 for voice notes and avatars.

Initialised lazily so the app starts even without AWS keys configured.
For MVP you can skip this entirely and store files elsewhere.
"""
import uuid
from app.core.config import settings

_s3_client = None


def _get_s3():
    """Get or create S3 client lazily — only when actually needed."""
    global _s3_client
    if _s3_client is None:
        import boto3
        kwargs = {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "region_name": settings.AWS_REGION,
        }
        # Only set endpoint_url if explicitly configured and non-empty
        if settings.S3_ENDPOINT_URL and settings.S3_ENDPOINT_URL.strip():
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL.strip()
        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def upload_audio(audio_bytes: bytes, folder: str = "voice-notes") -> str:
    """Upload audio bytes to S3/R2 and return public URL."""
    if not settings.AWS_ACCESS_KEY_ID:
        raise RuntimeError("AWS_ACCESS_KEY_ID not configured — set up S3/R2 or use Supabase Storage")

    key = f"{folder}/{uuid.uuid4()}.mp3"
    _get_s3().put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=key,
        Body=audio_bytes,
        ContentType="audio/mpeg",
    )
    if settings.S3_ENDPOINT_URL and settings.S3_ENDPOINT_URL.strip():
        return f"{settings.S3_ENDPOINT_URL.strip()}/{settings.S3_BUCKET_NAME}/{key}"
    return f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"


def get_presigned_upload_url(filename: str) -> dict:
    """Generate a presigned URL for direct client upload."""
    if not settings.AWS_ACCESS_KEY_ID:
        # Return a dummy response so profile setup doesn't break without S3
        return {
            "upload_url": "",
            "key": filename,
            "public_url": "",
        }

    key = f"uploads/{uuid.uuid4()}-{filename}"
    url = _get_s3().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.S3_BUCKET_NAME,
            "Key": key,
            "ContentType": "audio/m4a",
        },
        ExpiresIn=300,
    )
    public_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
    return {"upload_url": url, "key": key, "public_url": public_url}