#include <Python.h>
#include "numpy/arrayobject.h"


static char process_block_doc[] =
"process_block(data, threshold, hdt=1000, dead=1000, pos=0, limit=0, event=None, list=None)\n";

static PyObject *
process_block(PyObject* self, PyObject *args, PyObject *kwds)
{
    int threshold;
    long long hdt=1000, dead=1000, pos=0, limit=0;
    PyObject *data=NULL, *event_arg=NULL, *list=NULL;
    
    static char *kwlist[] = {"data", "threshold", "hdt", "dead", "pos", "limit", "event", "list", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi|LLLLOO",
                kwlist, &data, &threshold, &hdt, &dead, &pos, &limit, &event_arg, &list))
        return NULL;
    
    if (!PyArray_Check(data) || PyArray_TYPE((PyArrayObject *)data) != NPY_SHORT) {
        PyErr_SetString(PyExc_TypeError, "data must be array with 'i2' dtype");
        return NULL;
    }

    long long start, end;
    int event = 0;
    if (event_arg && event_arg != Py_None) {
        if ( !PyTuple_Check(event_arg) || !PyArg_ParseTuple(event_arg, "LL", &start, &end)) {
            PyErr_SetString(PyExc_TypeError, "event must be None or tuple of two integers");
            return NULL;
        }
        event = 1;
    }

    int new_list = 0; 
    if (list && list != Py_None){
        if ( !PyList_Check(list)){
            PyErr_SetString(PyExc_TypeError, "list must be None or list");
            return NULL;
        }
    } else {
        list = PyList_New(0);
        if (!list){
            return PyErr_NoMemory();
        }
        new_list = 1;
    }

#define EVENT(a,b) \
    { \
    PyObject *val = Py_BuildValue("(LL)", a, b); \
    if (!val) { \
        if (new_list) Py_DECREF(list); \
        Py_DECREF(iter); \
        return PyErr_NoMemory(); \
    } \
    PyList_Append(list, val); \
    Py_DECREF(val); \
    }
    

    PyObject *iter = PyArray_IterNew(data); 
    if (!iter) {
        if (new_list) Py_DECREF(list);
        return PyErr_NoMemory();
    }

    while (PyArray_ITER_NOTDONE(iter)) {
        const short d = *(short*)PyArray_ITER_DATA(iter);
        const long long i = ((PyArrayIterObject*)iter)->index + pos;
       
        if (d > threshold) {
            if (event){
                if (i < end-1+hdt) {
                    if (limit && end-start > limit){
                        /* emit and restart event if over limit */
                        EVENT(start, end);
                        start = i;
                        end = i+1;
                    } else {
                        /* extend event */
                        end = i+1;
                    }
                } else //this else makes 1s difference :)
                if (i >= end-1+hdt+dead) {
                    /* emit and restart event */
                    EVENT(start, end)
                    start = i;
                    end = i+1;
                }
            } else {
                /* start new event */
                start = i;
                end = i+1;
                event = 1;
            }
        }
        PyArray_ITER_NEXT(iter);
    }
    
    /* (possibly) emit last event */
    const long long i = ((PyArrayIterObject*)iter)->index + pos;
    if (event && i >= end-1+hdt+dead){
        EVENT(start, end)
        event = 0;
    }

    PyObject *ret;
    if (event)
        ret = Py_BuildValue("O(LL)", list, start, end);
    else
        ret = Py_BuildValue("OO", list, Py_None);

    Py_DECREF(iter);
    if (new_list) Py_DECREF(list);
    return ret;
}

static PyMethodDef event_detector_methods[] = {
    {"process_block", (PyCFunction)process_block, METH_VARARGS | METH_KEYWORDS, process_block_doc},
    {NULL}  /* Sentinel */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef event_detector = {
    "event_detector",
    NULL,
    -1,
    event_detector_methods
};
#endif

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC 
initevent_detector(void)
{
#if PY_VERSION_HEX >= 0x03000000
    PyModule_Create(&event_detector);
#else
    Py_InitModule("event_detector", event_detector_methods);
#endif
    import_array();
}
