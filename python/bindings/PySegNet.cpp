/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "PyTensorNet.h"
#include "PySegNet.h"

#include "segNet.h"

#include "../../utils/python/bindings/PyCUDA.h"

//-----------------------------------------------------------------------------------------
typedef struct {
	PyTensorNet_Object base;
	segNet* net;
} PySegNet_Object;


#define DOC_SEGNET "Image Segmenetation DNN - segments objects in an image\n\n" \
				  "Examples (jetson-inference/python/examples)\n" \
                      "     detectnet-console.py\n" \
				  "     detectnet-camera.py\n\n" \
				  "__init__(...)\n" \
				  "     Loads an object detection model.\n\n" \
				  "     Parameters:\n" \
				  "       network (string) -- name of a built-in network to use\n" \
				  "                           see below for available options.\n\n" \
				  "       argv (strings) -- command line arguments passed to imageNet,\n" \
				  "                         see below for available options.\n\n" \


// Init
static int PySegNet_Init( PySegNet_Object* self, PyObject *args, PyObject *kwds )
{
	printf(LOG_PY_INFERENCE "PySegNet_Init()\n");

	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "aerial-fpv";

	static char* kwlist[] = {"network", "argv", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sO", kwlist, &network, &argList))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ failed to parse args tuple");
		return -1;
	}

	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		printf(LOG_PY_INFERENCE "segNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "segNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);

			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ failed to parse argv list");
				return -1;
			}

			printf(LOG_PY_INFERENCE "segNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		self->net = segNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		printf(LOG_PY_INFERENCE "segNet loading build-in network '%s'\n", network);

		// parse the selected built-in network
		segNet::NetworkType networkType = segNet::NetworkTypeFromStr(network);

		if( networkType == segNet::SEGNET_CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid built-in network was requested");
			printf(LOG_PY_INFERENCE "detectNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}

		// load the built-in network
		self->net = segNet::Create(networkType);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet failed to load network");
		printf(LOG_PY_INFERENCE "segNet failed to load built-in network '%s'\n", network);
		return -1;
	}

	self->base.net = self->net;
	return 0;
}

// Detect
static PyObject* PySegNet_Process( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	static char* kwlist[] = {"image", "width", "height", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|", kwlist, &capsule, &width, &height))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to parse args tuple");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() image dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// run the object detection
	//detectNet::Detection* detections = NULL;

	//const int numDetections = self->net->Detect((float*)img, width, height, &detections, overlay > 0 ? detectNet::OVERLAY_BOX/*|detectNet::OVERLAY_LABEL*/ : detectNet::OVERLAY_NONE);
	const bool success = self->net->Process((float*)img, width, height);

	if( !success )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() encountered an error classifying the image");
		return NULL;
	}

	// create output objects
	PyObject* pyClass = PYLONG_FROM_LONG(2.0);

	// return tuple
	PyObject* tuple = PyTuple_Pack(1, pyClass);

	Py_DECREF(pyClass);

	return tuple;
}


// Detect
static PyObject* PySegNet_Overlay( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}

	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	static char* kwlist[] = {"image", "width", "height", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|", kwlist, &capsule, &width, &height))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to parse args tuple");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() image dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// run the object detection
	//detectNet::Detection* detections = NULL;

	//const int numDetections = self->net->Detect((float*)img, width, height, &detections, overlay > 0 ? detectNet::OVERLAY_BOX/*|detectNet::OVERLAY_LABEL*/ : detectNet::OVERLAY_NONE);
	const bool success = self->net->Overlay((float*)img, width, height);

	if( !success )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() encountered an error classifying the image");
		return NULL;
	}


	// create output objects
	PyObject* pyClass = PYLONG_FROM_LONG(2.0);

	// return tuple
	PyObject* tuple = PyTuple_Pack(1, pyClass);

	Py_DECREF(pyClass);

	return tuple;
}

//-------------------------------------------------------------------------------
static PyTypeObject pySegNet_Type =
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pySegNet_Methods[] =
{
	{ "Process", (PyCFunction)PySegNet_Process, METH_VARARGS|METH_KEYWORDS, DOC_SEGNET},
	{ "Overlay", (PyCFunction)PySegNet_Overlay, METH_VARARGS|METH_KEYWORDS, DOC_SEGNET},
	{NULL}  /* Sentinel */
};

// Register type
bool PySegNet_Register( PyObject* module )
{
	if( !module )
		return false;

	// skipping inner class registration

		/*
	 * register segNet type
	 */
	pySegNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".segNet";
	pySegNet_Type.tp_basicsize	= sizeof(PySegNet_Object);
	pySegNet_Type.tp_flags	= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pySegNet_Type.tp_base		= PyTensorNet_Type();
	pySegNet_Type.tp_methods	= pySegNet_Methods;
	pySegNet_Type.tp_new		= NULL; /*PySegNet_New;*/
	pySegNet_Type.tp_init		= (initproc)PySegNet_Init;
	pySegNet_Type.tp_dealloc	= NULL; /*(destructor)PySegNet_Dealloc;*/
	pySegNet_Type.tp_doc		= DOC_SEGNET;

	// complete registration of the detectNet type
	if( PyType_Ready(&pySegNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "segNet PyType_Ready() failed\n");
		return false;
	}

	Py_INCREF(&pySegNet_Type);

	if( PyModule_AddObject(module, "segNet", (PyObject*)&pySegNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "segNet PyModule_AddObject('segNet') failed\n");
		return false;
	}

	return true;
}
