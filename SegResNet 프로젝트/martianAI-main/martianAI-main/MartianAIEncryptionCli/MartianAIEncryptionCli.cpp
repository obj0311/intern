// MartianAIEncrptionCli.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <fstream>
#include "../MartianAI/Encryption.h"

using namespace std;

wchar_t* GetCmdOption(wchar_t** begin, wchar_t** end, const std::wstring& option)
{
	wchar_t** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool CmdOptionExists(wchar_t** begin, wchar_t** end, const std::wstring& option)
{
	return std::find(begin, end, option) != end;
}

int wmain(int argc, wchar_t **argv)
{
    wchar_t* infile;
	if (CmdOptionExists(argv, argv + argc, L"-f")) {
		infile = GetCmdOption(argv, argv + argc, L"-f");
	}
	else if (CmdOptionExists(argv, argv + argc, L"--file")) {
		infile = GetCmdOption(argv, argv + argc, L"--file");
	}
	else {
		throw std::invalid_argument("argument -f or --file is required");
	}
	char outfile[256] = { 0, };
	size_t filenameSize;
	wcstombs_s(&filenameSize,outfile , sizeof(outfile), infile, sizeof(outfile)-1);
	filenameSize -= 1; // NULL
	outfile[filenameSize]   = '.';
	outfile[filenameSize+1] = 'm';
	outfile[filenameSize+2] = 'i';
	outfile[filenameSize+3] = NULL;

	std::cout <<  "###########################################################\n";
	std::wcout << "# Input File:" << infile << "\n";
	std::cout <<  "# Output File:" << outfile << "\n";
	std::ifstream fin(infile, std::ios::binary);

	fin.seekg(0, ios::end);
	size_t length_load = fin.tellg();
	fin.seekg(0, ios::beg);
	unsigned char* modelBin = new unsigned char[length_load];
	fin.read(reinterpret_cast<char*>(modelBin), length_load * sizeof(unsigned char));

	unsigned char Key[] = { 0x76, 0x6f, 0x6c, 0x75, 0x6d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x20, 0x61, 0x69, 0x6c, 0x61, 0x62 };

	unsigned char expandedKey[176] = { 0, };

	KeyExpansion(Key, expandedKey);

	int encryptionSize = ENCRYPTION_SIZE;
	if (length_load < encryptionSize)
	{
		encryptionSize = 16 * (int)(length_load / 16);
	}

	for (int i = 0; i < encryptionSize / 2; i += 16) {
		AESEncrypt(modelBin + i, expandedKey, modelBin + i);
	}
	for (int i = length_load - 16; i > (length_load - encryptionSize / 2); i -= 16) {
		AESEncrypt(modelBin + i, expandedKey, modelBin + i);
	}

	std::ofstream FILE(outfile, std::ios::out | std::ofstream::binary);
	FILE.write(reinterpret_cast<const char*>(modelBin), length_load * sizeof(unsigned char));
	FILE.close();

	delete[] modelBin;

    std::cout << "Done!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
