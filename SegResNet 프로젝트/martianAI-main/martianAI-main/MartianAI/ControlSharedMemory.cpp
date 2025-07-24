#include "ControlSharedMemory.h"

ControlSharedMemory::ControlSharedMemory()
{
	size_t sizeInferenceInfo = sizeof(InferenceInfo);
	size_t sizeImageMemory = sizeof(DATA_TYPE) * MAX_DATA_SIZE;

	m_SizeSharedMemory = sizeInferenceInfo + sizeImageMemory;

	if (m_MapFile != NULL && m_SharedMemory != NULL)
	{
		m_SharedMemory = NULL;
		CloseHandle(m_MapFile);
		m_MapFile = NULL;
	}

	m_MapFile = CreateFileMappingA(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, static_cast<DWORD>(m_SizeSharedMemory), SHARED_MEMORY_NAME.c_str());
	m_SharedMemory = (BYTE*)MapViewOfFile(m_MapFile, FILE_MAP_ALL_ACCESS, 0, 0, m_SizeSharedMemory);

	m_InferenceInfoMemory = m_SharedMemory;
	m_DataMemory = m_SharedMemory + sizeInferenceInfo;
}

ControlSharedMemory::~ControlSharedMemory()
{
	if (m_SharedMemory != NULL)
	{
		m_DataMemory = NULL;
		m_InferenceInfoMemory = NULL;
		UnmapViewOfFile(m_SharedMemory);
		m_SharedMemory = NULL;
	}

	if (m_MapFile != NULL)
	{
		CloseHandle(m_MapFile);
		m_MapFile = NULL;
	}
}

bool ControlSharedMemory::ReadDataFromSharedMemory(DATA_TYPE * image, InferenceInfo& inferenceInfo)
{
	bool resultValue = false;

	if (ReceiveThroughSharedMemory<InferenceInfo>(&inferenceInfo, 1, m_InferenceInfoMemory))
	{
		int sendsize = inferenceInfo.xSize * inferenceInfo.ySize * inferenceInfo.zSize;

		if (sendsize > 0 && image != NULL)
		{
			resultValue = ReceiveThroughSharedMemory<DATA_TYPE>(image, sendsize, m_DataMemory);
		}
	}
	return resultValue;
}

bool ControlSharedMemory::WriteDataToSharedMemory(DATA_TYPE * image, InferenceInfo& inferenceInfo)
{
	bool resultValue = false;
	int sendsize = inferenceInfo.xSize * inferenceInfo.ySize * inferenceInfo.zSize;

	//resultValue = SendThroughSharedMemory<DATA_TYPE>(image, sendsize, m_DataMemory);
	//m_Width = imageInfo.width;
	//m_Height = imageInfo.height;

	if (SendThroughSharedMemory<InferenceInfo>(&inferenceInfo, 1, m_InferenceInfoMemory))
	{
		if (sendsize > 0 && image != NULL)
		{
			resultValue = SendThroughSharedMemory<DATA_TYPE>(image, sendsize, m_DataMemory);
		}
	}

	return resultValue;
}
