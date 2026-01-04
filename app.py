"""Entry point for the AI Document Comparison Gradio app."""
from __future__ import annotations

import os
import socket
from contextlib import closing
# Disable tokenizers parallelism to avoid fork warnings when using multiprocessing (e.g., Tesseract OCR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config.settings import settings
from utils.logging import configure_logging, logger
from visualization.gradio_ui import build_comparison_interface


configure_logging()


def _port_available(host: str, port: int) -> bool:
    if port == 0:
        return True

    def _can_bind(family: int, bind_host: str) -> bool:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as sock:
                sock.bind((bind_host, port))
                sock.listen(1)
            return True
        except OSError:
            return False

    # Check IPv4 explicitly.
    if host == "0.0.0.0":
        return _can_bind(socket.AF_INET, host)

    # Otherwise be conservative and check both families (where possible).
    ipv4_ok = _can_bind(socket.AF_INET, host)
    ipv6_ok = True
    try:
        ipv6_ok = _can_bind(socket.AF_INET6, "::")
    except Exception:
        ipv6_ok = True
    return ipv4_ok and ipv6_ok


def main() -> None:
    """Launch the Gradio application."""
    logger.info("Starting AI Document Comparison System")
    try:
        logger.info("Settings: model=%s, threshold=%.2f", 
                    settings.sentence_transformer_model, 
                    settings.text_similarity_threshold)
        logger.info("OCR engine: %s", settings.ocr_engine)
    except Exception:
        # Settings might not be fully initialized yet
        pass
    
    # Warmup OCR engines in background to avoid slow first request
    try:
        from extraction.ocr_router import warmup_ocr_engines
        warmup_ocr_engines(background=True)
        logger.info("OCR warmup started in background")
    except Exception as e:
        logger.warning("OCR warmup failed: %s", e)

    # Gradio's file inputs coming from MCP calls are often represented as URLs.
    # Gradio protects these downloads with SSRF checks that intentionally block
    # loopback/private hosts. For local development with our own Gradio server,
    # we opt-in to allowing loopback hosts so MCP can pass file URLs back to us.
    if os.getenv("DOC_DIFF_ALLOW_LOCAL_GRADIO_FILE_URLS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            import gradio.processing_utils as processing_utils

            for hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                if hostname not in processing_utils.PUBLIC_HOSTNAME_WHITELIST:
                    processing_utils.PUBLIC_HOSTNAME_WHITELIST.append(hostname)
            logger.warning(
                "Enabled local Gradio URL whitelist for MCP file inputs (dev-only)"
            )
        except Exception as e:
            logger.warning("Failed to patch Gradio SSRF whitelist: %s", e)
    
    interface = build_comparison_interface()

    # Port handling:
    # - Prefer explicit GRADIO_SERVER_PORT if provided.
    # - Otherwise try 7860 (default), and if busy fall back to the next ports.
    env_port = os.getenv("GRADIO_SERVER_PORT")
    preferred_port: int | None
    if env_port and env_port.strip():
        try:
            preferred_port = int(env_port)
        except ValueError:
            logger.warning("Invalid GRADIO_SERVER_PORT=%r; ignoring", env_port)
            preferred_port = None
    else:
        preferred_port = 7860

    port_candidates_raw = [preferred_port] if preferred_port is not None else []
    if preferred_port != 7860:
        port_candidates_raw.append(7860)
    port_candidates_raw.extend([7861, 7862, 7863, 7864, 7865])

    port_candidates: list[int] = []
    seen_ports: set[int] = set()
    for port in port_candidates_raw:
        if port in seen_ports:
            continue
        seen_ports.add(port)
        port_candidates.append(port)

    last_exc: Exception | None = None
    host = "0.0.0.0"
    for port in port_candidates:
        if not _port_available(host, port):
            logger.warning("Port %d already in use. Trying next...", port)
            continue
        try:
            logger.info("Launching Gradio on %s:%d", host, port)
            interface.launch(
                server_name=host,
                server_port=port,
                share=False,
                mcp_server=True,
            )
            return
        except OSError as exc:
            # Most common: [Errno 48] address already in use
            last_exc = exc
            logger.warning("Port %d unavailable (%s). Trying next...", port, exc)

    # Final fallback: let Gradio pick any free port.
    try:
        logger.info("No preferred ports free; letting Gradio choose a free port")
        interface.launch(
            server_name="0.0.0.0",
            server_port=0,
            share=False,
            mcp_server=True,
        )
        return
    except Exception as exc:
        if last_exc is not None:
            raise last_exc
        raise exc


if __name__ == "__main__":
    main()
