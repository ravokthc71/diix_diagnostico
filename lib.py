import os
import json
import time
import platform
import urllib.request
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HardwareDiagnostic:
    """Classe responsável por gerenciar todo o ciclo de diagnóstico e validação de hardware para IA."""

    def __init__(self):
        if not psutil:
            raise ImportError("A biblioteca 'psutil' é obrigatória. Instale com: pip install psutil")

    def _verificar_conexao_huggingface(self) -> bool:
        """Testa rapidamente se há acesso aos servidores do Hugging Face."""
        try:
            urllib.request.urlopen("https://huggingface.co", timeout=3)
            return True
        except Exception:
            return False

    def coletar_hardware(self) -> dict:
        """Coleta as informações brutas de hardware e realiza validações para IA."""
        # CPU
        cpu_info = {
            "processador": platform.processor() or platform.machine(),
            "nucleos_fisicos": psutil.cpu_count(logical=False),
            "nucleos_logicos": psutil.cpu_count(logical=True),
            "uso_cpu": psutil.cpu_percent(interval=0.5)
        }

        # Memória RAM
        ram = psutil.virtual_memory()
        ram_info = {
            "total_gb": round(ram.total / (1024 ** 3), 2),
            "disponivel_gb": round(ram.available / (1024 ** 3), 2),
            "uso_porcentagem": ram.percent
        }

        # Armazenamento (Disco principal)
        disk = psutil.disk_usage('/')
        espaco_livre_gb = round(disk.free / (1024 ** 3), 2)
        disk_info = {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "livre_gb": espaco_livre_gb,
            "uso_porcentagem": disk.percent,
            "suficiente_para_pesos": espaco_livre_gb >= 15.0  # Critério padrão de ~15GB para modelos de IA
        }

        # GPU / VRAM / CUDA
        cuda_info = {
            "disponivel": False,
            "quantidade": 0,
            "gpus": [],
            "aviso_configuracao": ""
        }

        if TORCH_AVAILABLE:
            cuda_info["disponivel"] = torch.cuda.is_available()
            if cuda_info["disponivel"]:
                cuda_info["quantidade"] = torch.cuda.device_count()
                for i in range(cuda_info["quantidade"]):
                    gpu_name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    vram_total = props.total_memory / (1024 ** 3)
                    
                    # Cálculo de VRAM livre e alocada atual
                    vram_alocada = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    vram_livre = vram_total - vram_alocada
                    
                    cuda_info["gpus"].append({
                        "id": i,
                        "nome": gpu_name,
                        "vram_total_gb": round(vram_total, 2),
                        "vram_livre_gb": round(vram_livre, 2)
                    })
            else:
                cuda_info["aviso_configuracao"] = "Atenção: A GPU não foi detectada pelo PyTorch. Verifique se o ambiente (ex: Kaggle) está configurado com Accelerator ativado (GPU T4x2 / P100) antes de continuar."
        else:
            cuda_info["aviso_configuracao"] = "PyTorch não instalado ou sem suporte CUDA."

        # Acesso Hugging Face
        hf_acesso = self._verificar_conexao_huggingface()

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": cpu_info,
            "ram": ram_info,
            "disk": disk_info,
            "cuda": cuda_info,
            "huggingface_acesso": hf_acesso
        }

    def processar_e_formatar(self) -> str:
        """Processa os dados brutos e retorna uma string formatada para exibição no console."""
        dados = self.coletar_hardware()
        
        linhas = []
        linhas.append("=" * 60)
        linhas.append("         RELATÓRIO DE DIAGNÓSTICO E VALIDAÇÃO DE IA         ")
        linhas.append("=" * 60)
        
        # Processador
        cpu = dados["cpu"]
        linhas.append(f"[+] Processador (CPU):")
        linhas.append(f"    - Modelo/Arquitetura : {cpu['processador']}")
        linhas.append(f"    - Núcleos Físicos    : {cpu['nucleos_fisicos']}")
        linhas.append(f"    - Núcleos Lógicos    : {cpu['nucleos_logicos']}")
        linhas.append(f"    - Uso Atual          : {cpu['uso_cpu']}%")
        
        # Memória RAM
        ram = dados["ram"]
        linhas.append(f"\n[+] Memória RAM:")
        linhas.append(f"    - Total              : {ram['total_gb']} GB")
        linhas.append(f"    - Disponível         : {ram['disponivel_gb']} GB")
        linhas.append(f"    - Uso                : {ram['uso_porcentagem']}%")
        
        # Disco & Validação de Pesos
        disk = dados["disk"]
        status_disco = "Sim (Suficiente)" if disk['suficiente_para_pesos'] else "Não (Abaixo de 15GB recomendados)"
        linhas.append(f"\n[+] Armazenamento (Disco Principal):")
        linhas.append(f"    - Total              : {disk['total_gb']} GB")
        linhas.append(f"    - Livre              : {disk['livre_gb']} GB")
        linhas.append(f"    - Espaço p/ Baixar   : {status_disco}")
        
        # CUDA / VRAM / GPU
        cuda = dados["cuda"]
        linhas.append(f"\n[+] Placa de Vídeo (GPU / VRAM / CUDA):")
        linhas.append(f"    - CUDA Disponível    : {'True' if cuda['disponivel'] else 'False'}")
        
        if cuda["disponivel"]:
            linhas.append(f"    - Qtd. de GPUs       : {cuda['quantidade']}")
            for g in cuda["gpus"]:
                linhas.append(f"      -> [{g['id']}] {g['nome']}")
                linhas.append(f"         VRAM Total      : {g['vram_total_gb']} GB")
                linhas.append(f"         VRAM Livre      : {g['vram_livre_gb']} GB (Disponível p/ Modelo/Contexto)")
        else:
            linhas.append(f"    - Alerta             : {cuda['aviso_configuracao']}")
            
        # Hugging Face
        hf_status = "Online (Acesso liberado)" if dados["huggingface_acesso"] else "Offline / Bloqueado"
        linhas.append(f"\n[+] Acesso ao Hugging Face:")
        linhas.append(f"    - Conectividade      : {hf_status}")
        
        linhas.append("=" * 60)
        return "\n".join(linhas)

    def gerar_json_completo(self) -> dict:
        """Consome os dados coletados e gera o JSON estruturado completo."""
        return self.coletar_hardware()

    def salvar_json_em_diretorio(self, nome_pasta: str = "diagnosticos") -> str:
        """Cria o diretório se não existir e salva o JSON com nome único, curto e identificável."""
        if not os.path.exists(nome_pasta):
            os.makedirs(nome_pasta)
            
        dados_json = self.gerar_json_completo()
        
        timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"diag_{timestamp_slug}.json"
        caminho_completo = os.path.join(nome_pasta, nome_arquivo)
        
        with open(caminho_completo, 'w', encoding='utf-8') as f:
            json.dump(dados_json, f, ensure_ascii=False, indent=4)
            
        return caminho_completo