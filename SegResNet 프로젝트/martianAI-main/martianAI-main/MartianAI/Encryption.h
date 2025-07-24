#pragma once
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>

#define ENCRYPTION_SIZE 4096000

// Encrypyion
void AESEncrypt(unsigned char* message, unsigned char* expandedKey, unsigned char* encryptedMessage);
void AddRoundKey(unsigned char* state, unsigned char* roundKey);
void SubBytesEnc(unsigned char* state);
void ShiftRowsEnc(unsigned char* state);
void MixColumns(unsigned char* state);
void RoundEnc(unsigned char* state, unsigned char* key);
void FinalRound(unsigned char* state, unsigned char* key);

// Decryption
void AESDecrypt(unsigned char* encryptedMessage, unsigned char* expandedKey, unsigned char* decryptedMessage);
void SubRoundKey(unsigned char* state, unsigned char* roundKey);
void InverseMixColumns(unsigned char* state);
void ShiftRowsDec(unsigned char* state);
void SubBytesDec(unsigned char* state);
void RoundDec(unsigned char* state, unsigned char* key);
void InitialRound(unsigned char* state, unsigned char* key);

// Common
void KeyExpansion(unsigned char inputKey[16], unsigned char expandedKeys[176]);
void KeyExpansionCore(unsigned char* in, unsigned char i);
