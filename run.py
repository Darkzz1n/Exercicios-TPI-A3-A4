import os
import fitz  # PyMuPDF
import numpy as np
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Carrega variáveis de ambiente
load_dotenv()

# Inicializa IA e o modelo Multilíngue (melhor para português)
ia_client = Groq()
gerador_embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =====================================================
# FUNÇÕES UTILITÁRIAS
# =====================================================

def calcular_similaridade(vetor_a, vetor_b):
    produto_escalar = np.dot(vetor_a, vetor_b)
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)
    return produto_escalar / (norma_a * norma_b)

def ler_arquivo_txt(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        return arquivo.read()

def ler_arquivo_pdf(caminho_arquivo):
    texto_completo = ""
    try:
        documento = fitz.open(caminho_arquivo)
        for pagina in documento:
            texto_completo += pagina.get_text()
        return texto_completo
    except Exception as e:
        return f"Erro ao ler PDF: {e}"

def criar_chunks(texto, tamanho_maximo=500, sobreposicao=50):
    """Divide um texto longo em pedaços menores com sobreposição."""
    lista_chunks = []
    passo = tamanho_maximo - sobreposicao
    
    for i in range(0, len(texto), passo):
        pedaco = texto[i : i + tamanho_maximo]
        lista_chunks.append(pedaco)
        if i + tamanho_maximo >= len(texto):
            break
            
    return lista_chunks

# =====================================================
# RESOLUÇÃO DOS EXERCÍCIOS
# =====================================================

def exercicio_1_e_2_ler_txt():
    print("\n--- EXTRAÇÃO E CHUNKING DE TXT ---")
    texto_txt = ler_arquivo_txt("docs/manual_clinica.txt")
    print("Texto extraído. Primeiros 100 caracteres:")
    print(texto_txt[:100] + "...\n")
    
    chunks = criar_chunks(texto_txt, 500)
    print(f"Quantidade de chunks gerados: {len(chunks)}")
    print("\nPrimeiro chunk:\n", chunks[0])
    print("\nÚltimo chunk:\n", chunks[-1])

def exercicio_1_e_2_ler_pdf():
    print("\n--- EXTRAÇÃO E CHUNKING DE PDF ---")
    texto_pdf = ler_arquivo_pdf("docs/artigo_saude.pdf")
    print("Texto extraído. Primeiros 100 caracteres:")
    print(texto_pdf[:100] + "...\n")
    
    chunks = criar_chunks(texto_pdf, 500)
    print(f"Quantidade de chunks gerados: {len(chunks)}")

def exercicio_3_gerar_embeddings():
    print("\n--- GERANDO EMBEDDINGS LOCAIS ---")
    frases_teste = [
        "A anatomia humana é composta por diversos sistemas interligados.",
        "O esmalte dentário é a substância mais dura do corpo humano.",
        "A higienização correta previne a proliferação de bactérias."
    ]
    
    vetores = gerador_embeddings.encode(frases_teste)
    print("Vetor da primeira frase gerado com sucesso!")
    print(f"Tamanho do vetor (dimensões): {len(vetores[0])}")
    print("Amostra dos 5 primeiros números do vetor:")
    print(vetores[0][:5])

def exercicio_4_busca_semantica():
    print("\n--- BUSCA SEMÂNTICA EM MEMÓRIA ---")
    conhecimento = [
        "O coração humano bate cerca de 100.000 vezes por dia.",
        "Os polvos possuem três corações e o sangue deles é azul.",
        "O fêmur é o osso mais longo e forte do corpo humano.",
        "Algumas espécies de bambu crescem até 90 centímetros em um único dia.",
        "A vitamina D é sintetizada pela pele através da exposição ao sol."
    ]
    
    vetores_conhecimento = gerador_embeddings.encode(conhecimento)
    
    pergunta_usuario = input("\nDigite sua pergunta: ")
    vetor_pergunta = gerador_embeddings.encode(pergunta_usuario)
    
    resultados = []
    for i, vetor_base in enumerate(vetores_conhecimento):
        pontuacao = calcular_similaridade(vetor_pergunta, vetor_base)
        resultados.append((pontuacao, conhecimento[i]))
        
    resultados.sort(reverse=True)
    
    print("\nFato mais relevante encontrado no banco:")
    print(f"-> {resultados[0][1]} (Score: {resultados[0][0]:.4f})")

def exercicio_5_rag_simples():
    print("\n--- RAG SIMPLES COM FATOS BIOLÓGICOS ---")
    conhecimento = [
        "Mel eterno: O mel é o único alimento que não estraga. Potes de 3.000 anos foram encontrados no Egito.",
        "Energia cerebral: O cérebro humano gera cerca de 20 watts de eletricidade quando está acordado.",
        "Impressões digitais: Assim como os humanos têm impressões digitais únicas, os cães têm focinhos únicos."
    ]
    
    vetores_conhecimento = gerador_embeddings.encode(conhecimento)
    pergunta_usuario = input("\nDigite sua pergunta: ")
    vetor_pergunta = gerador_embeddings.encode(pergunta_usuario)
    
    resultados = []
    for i, vetor_base in enumerate(vetores_conhecimento):
        pontuacao = calcular_similaridade(vetor_pergunta, vetor_base)
        resultados.append((pontuacao, conhecimento[i]))
        
    resultados.sort(reverse=True)
    contexto_recuperado = resultados[0][1]
    
    prompt_sistema = f"""
    Você é um assistente estrito e acadêmico. Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto abaixo.
    Se a resposta não estiver no contexto, diga: 'Não possuo essa informação.'
    
    CONTEXTO: {contexto_recuperado}
    PERGUNTA: {pergunta_usuario}
    """
    
    print("\nEnviando para a IA...")
    resposta_ia = ia_client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[{"role": "user", "content": prompt_sistema}]
    )
    
    print("\nRESPOSTA GERADA:")
    print(resposta_ia.choices[0].message.content)

def exercicio_6_pipeline_completo():
    print("\n--- PIPELINE RAG COMPLETO COM CHROMADB ---")
    
    print("1. Lendo arquivo e criando chunks...")
    texto_base = ler_arquivo_txt("docs/manual_clinica.txt")
    chunks_documento = criar_chunks(texto_base, 500)
    
    print("2. Gerando embeddings e salvando no Banco Vetorial...")
    vetores_documento = gerador_embeddings.encode(chunks_documento).tolist()
    
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    colecao = chroma_client.get_or_create_collection(
        name="manual_procedimentos",
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"chunk_{i}" for i in range(len(chunks_documento))]
    
    colecao.upsert(
        ids=ids,
        documents=chunks_documento,
        embeddings=vetores_documento
    )
    
    pergunta_usuario = input("\nFaça uma pergunta sobre os procedimentos da clínica: ")
    vetor_pergunta = gerador_embeddings.encode([pergunta_usuario]).tolist()
    
    # Trazendo os 4 melhores resultados para garantir o contexto completo
    resultados = colecao.query(
        query_embeddings=vetor_pergunta,
        n_results=4 
    )
    
    melhores_chunks = resultados["documents"][0]
    contexto_final = "\n\n---\n\n".join(melhores_chunks)
    
    prompt_final = f"""
    Atue como o Coordenador Clínico da Clínica Odontológica Bem-Estar.
    Baseado EXCLUSIVAMENTE no manual de procedimentos abaixo, responda à dúvida da equipe de forma clara e profissional.
    
    MANUAL (Contexto):
    {contexto_final}
    
    DÚVIDA DA EQUIPE:
    {pergunta_usuario}
    """
    
    print("\nGerando resposta baseada no manual (via LLM)...")
    resposta_ia = ia_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt_final}]
    )
    
    print("\nRESPOSTA DO COORDENADOR:")
    print(resposta_ia.choices[0].message.content)

# =====================================================
# MENU PRINCIPAL
# =====================================================

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("  PROJETO DE RAG - APLICAÇÃO CLÍNICA  ")
        print("="*40)
        print("1 - Ler e Dividir Arquivo TXT")
        print("2 - Ler e Dividir Arquivo PDF")
        print("3 - Testar Geração de Embeddings")
        print("4 - Busca Semântica em Memória")
        print("5 - Geração de Resposta com RAG Simples")
        print("6 - Executar Pipeline RAG Completo (ChromaDB)")
        print("0 - Sair")
        print("="*40)

        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            exercicio_1_e_2_ler_txt()
        elif escolha == "2":
            exercicio_1_e_2_ler_pdf()
        elif escolha == "3":
            exercicio_3_gerar_embeddings()
        elif escolha == "4":
            exercicio_4_busca_semantica()
        elif escolha == "5":
            exercicio_5_rag_simples()
        elif escolha == "6":
            exercicio_6_pipeline_completo()
        elif escolha == "0":
            print("Encerrando o programa...")
            break
        else:
            print("Opção inválida. Tente novamente.")