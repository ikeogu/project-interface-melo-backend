"""
Storage service — Supabase Storage (primary) or S3/R2 fallback.
"""
import uuid
import httpx
from app.core.config import settings

_s3_client = None


async def upload_audio(audio_bytes: bytes, folder: str = "voice-notes", content_type: str = "audio/mpeg") -> str:
    """Upload audio bytes and return public URL. Tries Supabase then S3."""
    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
        ext = "m4a" if "m4a" in content_type else "mp3"
        key = f"{folder}/{uuid.uuid4()}.{ext}"
        return await _upload_to_supabase(audio_bytes, key, content_type)

    if settings.AWS_ACCESS_KEY_ID:
        key = f"{folder}/{uuid.uuid4()}.mp3"
        return _upload_to_s3(audio_bytes, key)

    raise RuntimeError("No storage configured — set SUPABASE_URL/SUPABASE_SERVICE_KEY or AWS credentials")


async def _upload_to_supabase(audio_bytes: bytes, key: str, content_type: str) -> str:
    bucket = "voice-notes"
    supabase_url = settings.SUPABASE_URL.rstrip("/")

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{supabase_url}/storage/v1/object/{bucket}/{key}",
            headers={
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                "Content-Type": content_type,
                "x-upsert": "true",
            },
            content=audio_bytes,
        )

        if r.status_code not in (200, 201):
            await _ensure_bucket(client, bucket, supabase_url)
            r = await client.post(
                f"{supabase_url}/storage/v1/object/{bucket}/{key}",
                headers={
                    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
                    "Content-Type": content_type,
                    "x-upsert": "true",
                },
                content=audio_bytes,
            )
            if r.status_code not in (200, 201):
                raise RuntimeError(f"Supabase upload failed: {r.text}")

    return f"{supabase_url}/storage/v1/object/public/{bucket}/{key}"


async def _ensure_bucket(client: httpx.AsyncClient, bucket: str, supabase_url: str) -> None:
    await client.post(
        f"{supabase_url}/storage/v1/bucket",
        headers={
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        },
        json={"id": bucket, "name": bucket, "public": True},
    )


def _upload_to_s3(audio_bytes: bytes, key: str) -> str:
    global _s3_client
    if _s3_client is None:
        import boto3
        kwargs = {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "region_name": settings.AWS_REGION,
        }
        if settings.S3_ENDPOINT_URL and settings.S3_ENDPOINT_URL.strip():
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL.strip()
        _s3_client = boto3.client("s3", **kwargs)

    _s3_client.put_object(
        Bucket=settings.S3_BUCKET_NAME,
        Key=key,
        Body=audio_bytes,
        ContentType="audio/mpeg",
    )
    if settings.S3_ENDPOINT_URL and settings.S3_ENDPOINT_URL.strip():
        return f"{settings.S3_ENDPOINT_URL.strip()}/{settings.S3_BUCKET_NAME}/{key}"
    return f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"


def get_presigned_upload_url(filename: str) -> dict:
    """Generate a presigned S3 URL for direct client upload."""
    if not settings.AWS_ACCESS_KEY_ID:
        return {"upload_url": "", "key": filename, "public_url": ""}

    key = f"uploads/{uuid.uuid4()}-{filename}"
    url = _upload_to_s3.__globals__  # ensure client init — call _upload_to_s3 side-effects via lazy init
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key, "ContentType": "audio/m4a"},
        ExpiresIn=300,
    )
    public_url = f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
    return {"upload_url": url, "key": key, "public_url": public_url}
