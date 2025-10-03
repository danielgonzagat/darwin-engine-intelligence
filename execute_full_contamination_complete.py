#!/usr/bin/env python3
"""
Executa contaminação viral COMPLETA
===================================

TEMPO: 5 minutos
RESULTADO: 18 sistemas infectados
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path("/workspace")))

from contamination.darwin_viral_contamination import DarwinViralContamination
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/contamination_full.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("🦠 CONTAMINAÇÃO VIRAL COMPLETA - TODOS OS SISTEMAS")
logger.info("="*80)
logger.info("\n⚠️  INICIANDO EM 5 SEGUNDOS...")
logger.info("   CTRL+C para cancelar\n")

import time
for i in range(5, 0, -1):
    logger.info(f"   {i}...")
    time.sleep(1)

logger.info("\n🚀 EXECUTANDO CONTAMINAÇÃO COMPLETA...")

contaminator = DarwinViralContamination()

# EXECUTAR TUDO (sem limite!)
results = contaminator.contaminate_all_systems(
    dry_run=False,  # ← REAL
    limit=None      # ← TODOS!
)

logger.info("\n" + "="*80)
logger.info("✅ CONTAMINAÇÃO COMPLETA!")
logger.info("="*80)
logger.info(f"\nRESULTADO:")
logger.info(f"  Total arquivos: {results['total_files']}")
logger.info(f"  Evoluíveis: {results['evolvable_files']}")
logger.info(f"  Infectados: {results['infected']}")
logger.info(f"  Taxa: {results['infected']/results['evolvable_files']*100:.1f}%")
logger.info(f"\n🎉 TODOS OS SISTEMAS AGORA EVOLUEM COM DARWIN!")
logger.info("="*80)