#ifndef VOCABHASH_H
#define VOCABHASH_H

#include "uthash.h"

// Structure for vocabulary hash table entries
typedef struct {
    char *word;             // Key: the vocabulary word
    UT_hash_handle hh;     // Makes this structure hashable by uthash
} VocabEntry;

#endif