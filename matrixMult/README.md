# Resultados

`I1`: memoria caché de instrucciones de primer nivel.  
`D1`: memoria caché de datos de primer nivel.  
`LL`: memoria caché de último nivel (L2 o L3 dependiendo del sistema).  

# Multiplicación de matrices normal:
```
==3469== I   refs:      7,019,239,891
==3469== I1  misses:            1,434
==3469== LLi misses:            1,393
==3469== I1  miss rate:          0.00%
==3469== LLi miss rate:          0.00%
==3469== 
==3469== D   refs:      2,004,051,940  (2,000,752,907 rd   + 3,299,033 wr)
==3469== D1  misses:    1,064,151,322  (1,063,022,410 rd   + 1,128,912 wr)
==3469== LLd misses:       62,753,740  (   62,563,220 rd   +   190,520 wr)
==3469== D1  miss rate:          53.1% (         53.1%     +      34.2%  )
==3469== LLd miss rate:           3.1% (          3.1%     +       5.8%  )
==3469== 
==3469== LL refs:       1,064,152,756  (1,063,023,844 rd   + 1,128,912 wr)
==3469== LL misses:        62,755,133  (   62,564,613 rd   +   190,520 wr)
==3469== LL miss rate:            0.7% (          0.7%     +       5.8%  )
```

En `I1` hubo muy poca cantidad de misses, debido a que el programa es pequeño.  
En `D1` hubo una gran cantidad de misses (53.1% de miss rate), es lo esperado
debido a la técnica usada. Los valores de la matriz se tienen que cargar en 
caché repetidas veces.  
En `LL` la cantidad de misses fue pequeña (0.7% de miss rate).

# Multiplicación de matrices por bloques:

```
==3421== I   refs:      7,681,050,118
==3421== I1  misses:            1,443
==3421== LLi misses:            1,402
==3421== I1  miss rate:          0.00%
==3421== LLi miss rate:          0.00%
==3421== 
==3421== D   refs:      2,077,998,111  (2,075,694,912 rd   + 2,303,199 wr)
==3421== D1  misses:       13,585,116  (   13,456,209 rd   +   128,907 wr)
==3421== LLd misses:        4,231,274  (    4,103,494 rd   +   127,780 wr)
==3421== D1  miss rate:           0.7% (          0.6%     +       5.6%  )
==3421== LLd miss rate:           0.2% (          0.2%     +       5.5%  )
==3421== 
==3421== LL refs:          13,586,559  (   13,457,652 rd   +   128,907 wr)
==3421== LL misses:         4,232,676  (    4,104,896 rd   +   127,780 wr)
==3421== LL miss rate:            0.0% (          0.0%     +       5.5%  )

```

En `I1` hubo muy poca cantidad de misses, debido a que el programa es pequeño.  
En `D1` hubo una pequeña cantidad de misses en comparación al primer método
(0.7% de miss rate), usando bloques se hace un mejor uso de la memoria caché, ya
que se usan los datos que actualmente están en memoria caché (se cargan bloques
pequeños).  
En `LL` la cantidad de misses fue mínima.
