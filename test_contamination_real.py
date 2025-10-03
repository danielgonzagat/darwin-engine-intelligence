#!/usr/bin/env python3
"""
Teste de Contaminação Real
==========================

Executa contaminação viral REAL em alguns arquivos
para demonstrar o funcionamento.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path("/workspace")))

from contamination.darwin_viral_contamination import DarwinViralContamination
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_contamination():
    """Testa contaminação real"""
    logger.info("="*80)
    logger.info("🦠 TESTE DE CONTAMINAÇÃO REAL")
    logger.info("="*80)
    
    contaminator = DarwinViralContamination()
    
    # Contaminar alguns arquivos de verdade
    logger.info("\n🚀 Executando contaminação REAL (limit=10)...")
    results = contaminator.contaminate_all_systems(dry_run=False, limit=10)
    
    logger.info("\n📊 RESULTADOS:")
    logger.info(f"   Total arquivos: {results['total_files']}")
    logger.info(f"   Evoluíveis: {results['evolvable_files']}")
    logger.info(f"   Infectados: {results['infected']}")
    logger.info(f"   Falhados: {results['failed']}")
    logger.info(f"   Taxa sucesso: {results['infected']/(results['infected']+results['failed'])*100:.1f}%")
    
    # Verificar arquivos infectados
    logger.info("\n🔍 Verificando arquivos infectados:")
    infected_files = list(Path("/workspace").glob("*_DARWIN_INFECTED.py"))
    logger.info(f"   Arquivos *_DARWIN_INFECTED.py encontrados: {len(infected_files)}")
    
    for file_path in infected_files[:3]:  # Mostrar apenas os 3 primeiros
        logger.info(f"   ✅ {file_path.name}")
    
    if len(infected_files) > 3:
        logger.info(f"   ... e mais {len(infected_files) - 3} arquivos")
    
    logger.info("\n" + "="*80)
    logger.info("✅ CONTAMINAÇÃO REAL COMPLETA!")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    results = test_real_contamination()