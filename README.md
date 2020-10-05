# Filtro de Kalman para Pair Trading

Função para aplicação do Filtro de Kalman para operações de Pair Trading (Long&Short)

Basicamente um radar para localizar operações.

Primeiro filtramos os pares que são cointegrados entre si, e para esses pares, é aplicado o filtro de kalman.

Com base no z-Score é passado um radar em busca dos pares que estão dando entrada.

Código ainda em fase de desenvolvimento, muita coisa precisa ser melhorada/otimizada.

Para base de dados verificar: https://github.com/rodrigoaugustov/cotacoes_tryd 

De preferência limitar o período máximo de dados.
