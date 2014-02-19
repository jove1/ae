#include <Python.h>
#include "numpy/arrayobject.h"


static char process_block_doc[] =
"process_block(data, threshold, hdt=1000, dead=1000, pos=0, event=None, list=None)\n";

static PyObject *
process_block(PyObject* self, PyObject *args, PyObject *kwds)
{
    int threshold, hdt = 1000, dead = 1000, pos=0;
    PyObject *data = NULL, *event_arg = NULL, *list=NULL;
    
    static char *kwlist[] = {"data", "threshold", "hdt", "dead", "pos", "event", "list", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi|iiiOO", kwlist, &data, &threshold, &hdt, &dead, &pos, &event_arg, &list))
        return NULL;
    
    if (!PyArray_Check(data) || PyArray_TYPE(data) != PyArray_SHORT) {
        PyErr_SetString(PyExc_TypeError, "data must be array with 'i2' dtype");
        return NULL;
    }

    int start, end, event = 0;
    if (event_arg && event_arg != Py_None) {
        if ( !PyTuple_Check(event_arg) || !PyArg_ParseTuple(event_arg, "ii", &start, &end)) {
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
    PyObject *val = Py_BuildValue("(ii)", a, b); \
    if (!val) { \
        if (new_list) Py_DECREF(list); \
        Py_DECREF(iter); \
        return PyErr_NoMemory(); \
    } \
    PyList_Append(list, val); \
    Py_DECREF(val);
    

    PyObject *iter = PyArray_IterNew(data); 
    if (!iter) {
        if (new_list) Py_DECREF(list);
        return PyErr_NoMemory();
    }

    while (PyArray_ITER_NOTDONE(iter)) {
        const short d = *(short*)PyArray_ITER_DATA(iter);
        const int i = ((PyArrayIterObject*)iter)->index + pos;
       
        if (d > threshold) {
            if (event){
                if (i < end-1+hdt) {
                    /* extend event */
                    end = i+1;
                } else //this else makes 1s difference :)
                if (i >= end-1+hdt+dead) {
                    /* emit and restart event*/
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
    const int i = ((PyArrayIterObject*)iter)->index + pos;
    if (event && i >= end-1+hdt+dead){
        EVENT(start, end)
        event = 0;
    }


    if (event)
        return Py_BuildValue("O(ii)", list, start, end);
    else
        return Py_BuildValue("OO", list, Py_None);
}

static PyMethodDef event_detector_methods[] = {
    {"process_block", (PyCFunction)process_block, METH_VARARGS | METH_KEYWORDS, process_block_doc},
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC 
initevent_detector(void)
{
    Py_InitModule("event_detector", event_detector_methods);
    import_array();
}
