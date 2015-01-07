

all: hw1

hw1: main.o merge.o wordll.o word_tokenize.o
	gcc main.o merge.o wordll.o word_tokenize.o -pthread -o hw1

main.o: main.c
	gcc -c -Wall -pthread main.c

merge.o: merge.c
	gcc -c -Wall -pthread merge.c

wordll.o: wordll.c
	gcc -c -Wall -pthread wordll.c

word_tokenize.o: word_tokenize.c
	gcc -c -Wall -pthread word_tokenize.c

clean:
	rm -rf *o hw1
