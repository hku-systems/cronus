/*! \file   CudaWorkerThread.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday October 23, 2011
	\brief  The source file for the CudaWorkerThread class.
*/

// Ocelot Includes
#include <ocelot/cuda/interface/CudaWorkerThreadFake.h>
#include <ocelot/executive/interface/Device.h>

// Hydrazine Includes
#include <hydrazine/interface/Exception.h>
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace cuda
{

class WorkerMessage
{
public:
	enum Type
	{
		Kill,
		Wait,
		Launch,
		Invalid
	};
	
public:
	Type                      type;
	CudaWorkerThread::Launch* launch;
};


CudaWorkerThread::CudaWorkerThread()
: _launched(0), _finished(0), _device(0)
{

}

CudaWorkerThread::~CudaWorkerThread()
{
	WorkerMessage message;
	message.type = WorkerMessage::Kill;
	execute(&message);
}

void CudaWorkerThread::setDevice(executive::Device* d)
{
	_device = d;
}

void CudaWorkerThread::launch(const std::string& module, 
	const std::string& kernel, const ir::Dim3& grid, 
	const ir::Dim3& block, size_t sharedMemory, 
	const void* argumentBlock, size_t argumentBlockSize/*, 
	const trace::TraceGeneratorVector& traceGenerators,
	const ir::ExternalFunctionSet* externals*/)
{
	assert(_device != 0);
	report("Launching kernel '" << kernel << "' executing on device '"
		<< _device->properties().name << "'");
	++_launched;

	Launch* l = new Launch;

	l->module       = module;
	l->kernel       = kernel;
	l->gridDim      = grid;
	l->blockDim     = block;
	l->sharedMemory = sharedMemory;
//	l->generators   = traceGenerators;
//	l->externals    = externals;

	l->parameters.insert(l->parameters.end(),
		(const char*)argumentBlock,
		(const char*)argumentBlock + argumentBlockSize);

	WorkerMessage* message = new WorkerMessage;
	message->type   = WorkerMessage::Launch;
	message->launch = l;

	report(" sending message " << message);
	execute(message);
}

void CudaWorkerThread::wait()
{
	if(!areAnyKernelsRunning()) return;

	assert(_device != 0);
	report("Waiting for all kernels to finish executing on device '"
		<< _device->properties().name << "'");

	WorkerMessage message;
	message.type = WorkerMessage::Wait;

	execute(&message);
}

bool CudaWorkerThread::areAnyKernelsRunning()
{
	return _launched - _finished > 0;
}

void CudaWorkerThread::execute() {}

void CudaWorkerThread::execute(void* msg)
{
	WorkerMessage* message = (WorkerMessage*)msg;
	try
	{
		switch(message->type)
		{
		case WorkerMessage::Wait:
		{
			while(!_launches.empty())
			{
				_launchNext();
			}

			break;
		}
		case WorkerMessage::Launch:
		{
			report(" Received kernel launch message.");
			_launches.push(*message->launch);

			delete message->launch;
			delete message;
			break;
		}
		default: assertM(false, "Invalid message type.");
		}
	}
	catch(const hydrazine::Exception& e)
	{
		report("Operation failed, replying with exception.");
		assertM(false, "Multi-threaded exeptions not supported.");
	}
	
	if(!_launches.empty())
	{
		_launchNext();
	}
		
}

void CudaWorkerThread::_launchNext()
{
	assert(_device != 0);
	
	Launch& l = _launches.front();
	
	report(" Launching kernel '" << l.kernel << "' now.");

	_device->select();
	_device->launch(l.module, l.kernel, l.gridDim, l.blockDim, l.sharedMemory, 
		l.parameters.data(), l.parameters.size()/*, l.generators, l.externals*/);
	_device->unselect();
	
	report("  kernel '" << l.kernel << "' finished.");
	
	++_finished;
	
	_launches.pop();
}

}

