CC = gcc
CFLAGS = -Wall -g  -I./include # -Wall enables warnings, -g adds debugging info

main: main.o network.o dataParser.o
	$(CC) $(CFLAGS) -o main main.o network.o dataParser.o

main.o: main.c ./network/network.h ./dataParsing/dataParser.h ./dataParsing/vocabHash.h
	$(CC) $(CFLAGS) -c main.c

network.o: ./network/network.c ./network/network.h
	$(CC) $(CFLAGS) -c ./network/network.c

dataParser.o: ./dataParsing/dataParser.c ./dataParsing/dataParser.h
	$(CC) $(CFLAGS) -c ./dataParsing/dataParser.c

clean:
	rm -f *.o main
