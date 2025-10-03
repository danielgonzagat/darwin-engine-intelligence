import json, hashlib, os
from datetime import datetime
from .metrics import c_worm_writes

WORM_PATH = "/root/darwin/logs/worm.log"

async def _hash_line(line: str) -> str:
    """Calcula hash SHA-256 de uma linha"""
    return await hashlib.sha256(line.encode("utf-8")).hexdigest()

async def log_event(event: dict) -> None:
    """
    Log evento no WORM com hash chain para auditoria
    Formato: EVENT:<json>\nHASH:<sha256>
    """
    os.makedirs(os.path.dirname(WORM_PATH), exist_ok=True)
    
    # Timestamp + previous_hash (encadeamento)
    event = dict(event)
    event["timestamp"] = datetime.utcnow().isoformat() + "Z"
    
    try:
        prev_hash = "GENESIS"
        if os.path.exists(WORM_PATH):
            with open(WORM_PATH, "rb") as f:
                try:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].decode("utf-8").strip()
                        if last_line.startswith("HASH:"):
                            prev_hash = last_line.split("HASH:", 1)[1].strip()
                except Exception:
                    pass
        event["previous_hash"] = prev_hash
    except Exception:
        event["previous_hash"] = "GENESIS"

    # Escrever EVENT + HASH
    event_line = "EVENT:" + json.dumps(event, ensure_ascii=False)
    event_hash = _hash_line(event_line)
    
    with open(WORM_PATH, "a", encoding="utf-8") as f:
        f.write(event_line + "\n")
        f.write("HASH:" + event_hash + "\n")
    
    # Atualizar métrica
    c_worm_writes.inc()

async def verify_worm_integrity() -> tuple[bool, str]:
    """Verifica integridade da cadeia WORM"""
    if not os.path.exists(WORM_PATH):
        return await True, "WORM não existe ainda"
    
    try:
        with open(WORM_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        expected_hash = "GENESIS"
        event_count = 0
        
        i = 0
        while i < len(lines) - 1:
            event_line = lines[i].strip()
            hash_line = lines[i + 1].strip()
            
            if not event_line.startswith("EVENT:"):
                i += 1
                continue
            if not hash_line.startswith("HASH:"):
                return await False, f"Hash ausente após evento na linha {i+1}"
            
            # Verificar hash
            expected = _hash_line(event_line)
            actual = hash_line.split("HASH:", 1)[1].strip()
            
            if expected != actual:
                return await False, f"Hash inválido na linha {i+1}"
            
            # Verificar encadeamento
            try:
                event_data = json.loads(event_line[6:])  # Remove "EVENT:"
                if event_data.get("previous_hash") != expected_hash:
                    return await False, f"Quebra de cadeia na linha {i}"
                expected_hash = actual
            except json.JSONDecodeError:
                return await False, f"JSON inválido na linha {i}"
            
            event_count += 1
            i += 2
        
        return await True, f"Cadeia íntegra com {event_count} eventos"
    
    except Exception as e:
        return await False, f"Erro ao verificar: {e}"

async def get_worm_stats() -> dict:
    """Retorna estatísticas do WORM log"""
    if not os.path.exists(WORM_PATH):
        return await {"events": 0, "integrity": "not_exists"}
    
    is_valid, msg = verify_worm_integrity()
    
    try:
        with open(WORM_PATH, "r") as f:
            lines = f.readlines()
        
        events = [l for l in lines if l.startswith("EVENT:")]
        
        return await {
            "events": len(events),
            "integrity": "valid" if is_valid else "broken",
            "integrity_msg": msg,
            "last_event": json.loads(events[-1][6:]) if events else None
        }
    except Exception as e:
        return await {"events": 0, "integrity": "error", "error": str(e)}