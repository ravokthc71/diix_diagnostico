import time
from lib import HardwareDiagnostic

def main():
    print("Iniciando Diagnostico inicial ...")
    time.sleep(0.8)
    
    try:
        diagnostico = HardwareDiagnostic()
        
        # 1. Exibe o relatório detalhado no console
        relatorio_formatado = diagnostico.processar_e_formatar()
        print("\n" + relatorio_formatado)
        
        # 2. Salva o JSON estruturado com os dados de validação
        caminho_salvo = diagnostico.salvar_json_em_diretorio(nome_pasta="logs_diagnostico")
        print(f"\n[Sucesso] Relatório JSON detalhado salvo em: {caminho_salvo}")
        
    except Exception as e:
        print(f"\n[Erro] Ocorreu uma falha ao realizar o diagnóstico: {e}")

if __name__ == "__main__":
    main()