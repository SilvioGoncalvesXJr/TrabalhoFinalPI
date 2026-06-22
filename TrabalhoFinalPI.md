

Trabalho Final – Processamento de Imagens
## Prof. Matheus Araújo
## 1. Descrição
## Geral:
Desenvolver um projeto aplicado de Processamento Digital de Imagens em formato
de artigo, utilizando técnicas estudadas ao longo da disciplina para resolver um
problema relacionado a um dos temas propostos.
## Detalhes:
O projeto deve contemplar um pipeline de processamento de imagens clássico ou de
aprendizado envolvendo etapas como pré-processamento, transformação, análise,
treinamento e interpretação dos resultados. Espera-se que o trabalho inclua a
implementação manual (sem uso de funções prontas para as etapas principais) de
técnicas abordadas em aula, tais como:
- Transformações no domínio espacial (ex: filtros, operações pontuais);
- Transformações no domínio da frequência (ex: DFT/DCT – ao menos
conceitualmente ou simplificado);
- Técnicas de compressão (ex: quantização, DCT, Huffman ou análise de
entropia);
- Representação e descrição de imagens (ex: histogramas, estatísticas,
descritores);
- Segmentação de imagens e Detecção de Bordas (ex: limiarização, bordas,
morfologia, watershed ou similares).
- Aplicação de uma técnica de aprendizado de máquina, como classificação ou
agrupamento (ex: KNN, K-means, regressão logística ou métodos simples),
utilizando características extraídas das imagens para análise ou tomada de
decisão.
O artigo deve ser aplicado a um contexto real dentro do tema escolhido, com uso de
imagens coerentes com a proposta.

## 2. Temas
Cada grupo escolheu um dos temas abaixo na dinâmica do classroom:
- Medicina (Exames)
## 2. Astronomia
- Biologia (Microscopia)
- Sensoriamento Remoto (Mapeamentos Aéreos)
- Animais e Natureza (Preferência: Gatos)

- Indústria e Inspeção (Equipamentos)
- Arte (Pinturas/Esculturas)
- Veículos Autônomos (Trânsito Urbano)
## 9. Gastronomia
- Arquitetura e Prédios
- Carros e Motos
## 12. Ambientes Domésticos
## 13. Ambientes Aquáticos
- Moda (Roupas/Modelos)
## 15. Memes

## 3. Requisitos Técnicos
- É permitido o uso de bibliotecas como OpenCV apenas para leitura e escrita
de imagens e operações básicas;
- As principais técnicas devem ser implementadas manualmente;
- As técnicas de aprendizado de máquina podem ser implementadas usando
bibliotecas como scikit-learn, tensorflow ou pytorch.
- O projeto deve conter, obrigatoriamente:
o Escolha de algum dataset relacionado ao seu tema, tendo a opção de
geração manual de dataset ou com ferramentas automáticas.
o Pelo menos 1 técnica de transformação de imagem (espacial OU na
frequência);
o Pelo menos 1 técnica de compressão (DCT, Hoffman etc.) OU análise de
descritiva da informação (entropia, assimetria, curtose, etc.);
o Uma das duas aplicações abaixo:
Pipeline de Processamento Clássico: pelo menos 1 técnica de morfologia e
1 técnica de segmentação ou detecção de bordas.
## OU
Pipeline de Aprendizado Visual: pelo menos 1 técnica de aprendizado de
máquina em imagens (SVM, KNN, K-MEANS, CNN etc).
- As operações devem ser realizadas para todas as imagens do dataset.
- Após as transformações de pré-processamento as imagens devem seguir para um
modelo de pi aprendizado de máquina.
- O sistema deve apresentar resultados visuais e quantitativos;
- Deve haver análise e interpretação dos resultados obtidos.
- No caso de Pipeline de Aprendizado Visual a escolha de hiperparâmetros bem
como o estudo de variação dos mesmos devem ser explicados no artigo. Os
modelos utilizados também devem ter descrição e justificativa.






## 4. Equipe
O trabalho deve ser desenvolvido em grupos de 3 a 5 alunos.

## 5. Pontuação
O trabalho corresponde à nota da avaliação e vale de 0 a 10,0.
- 40% – Implementação técnica:
o Correta aplicação dos algoritmos;
o Implementações manuais;
o Funcionamento geral do pipeline;
- 20% – Aplicação e criatividade:
o Adequação ao tema escolhido;
o Qualidade dos experimentos;
o Originalidade na proposta;
- 15% – Documentação e organização:
o Clareza do código;
o Organização do repositório;
o Qualidade do artigo;
## • 25% – Apresentação:
o Clareza na explicação;
o Demonstração dos resultados;
o Domínio do conteúdo.

## 6. Entrega
- Data limite: 21/06
- A entrega será realizada via Google Classroom, contendo:
O repositório deve incluir:
a) Código-fonte completo do projeto;
b) README contendo:
- Descrição do problema abordado;
- Técnicas utilizadas;
- Instruções de execução;
c) Link para o dataset ou conjunto de imagens utilizadas nos testes;


Além disso, deve ser entregue:
- Artigo em PDF no formato SBC, contendo:
o Introdução ao problema;
o Metodologia;
o Resultados (com imagens);
o Análises e interpretações;
o Conclusão.
Template LaTeX: https://www.overleaf.com/latex/templates/sbc-
conferences-template-updated-sbc-template-dot-sty-v2017/pyhttxftxjqn
- Slides da apresentação (formato livre – PDF ou PowerPoint).

## 7. Observações Gerais
- As imagens utilizadas devem estar relacionadas ao tema escolhido;
- Não é permitido o uso de funções prontas que resolvam diretamente os
problemas propostos;
- O foco do trabalho é o entendimento e implementação dos métodos;
- Projetos que utilizarem técnicas além do conteúdo visto em sala poderão receber
destaque positivo na avaliação.

## 8. Bônus
Projetos que apresentarem:
- Interface interativa (dashboard hospedado);
- Comparação entre múltiplas técnicas;
- Aplicações mais completas (ex: mini-sistema funcional);
poderão receber até 1 ponto extra, a critério do professor.
