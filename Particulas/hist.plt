
set xlabel "Velocidade"
set ylabel "Frequência da velocidade"

set auto x
set auto y
set style fill solid border -1
set boxwidth 0.9 rel
set xtic rotate by -45 scale 0
#set bmargin 10 


#defining a function
f(x)= (a/(2*3.1415926*s**2)**0.5*exp(-(x-m)**2/(2*s)))
a = 10000
s = 1.0
m = 1.0

set dummy x	#setting the dummy variable

#defining a line style
set style line 1 lc rgb'#00EEDD' lt 1 lw 1 

fit f(x) 'hist.dat' using 1:2 via a,s,m


plot "hist.dat" using 1:2 title '' w boxes, f(x)
