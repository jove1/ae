#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    PyObject *callback;
    int event_start;
    int last;
} EventDetector;

static void EventDetector_dealloc(EventDetector* self) {
    Py_XDECREF(self->callback);
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *EventDetector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    EventDetector *self;
    self = (EventDetector *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->event_start = -1;
        self->last = -1;
        self->callback = NULL;
    }
    return (PyObject *)self;
}

static int EventDetector_init(EventDetector *self, PyObject *args, PyObject *kwds) {
    PyObject *callback=NULL, *tmp;
    
    static char *kwlist[] = {"callback", "event_start", "last", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O|ii", kwlist, &callback, &self->event_start, &self->last))
        return -1; 

    if (!PyCallable_Check(callback)) {
        PyErr_SetString(PyExc_TypeError, "callback must be callable");
        return -1;
    }

    tmp = self->callback;
    Py_INCREF(callback);
    self->callback = callback;
    Py_XDECREF(tmp);

    return 0;
}

static PyMemberDef EventDetector_members[] = {
    {"callback", T_OBJECT_EX, offsetof(EventDetector, callback), 0, "callback"},
    {"event_start", T_INT, offsetof(EventDetector, event_start), 0, "event_start"},
    {"last", T_INT, offsetof(EventDetector, last), 0, "last"},
    {NULL}  /* Sentinel */
};


static inline void func(int i, short d, int threshold, int hdt, int dead, EventDetector* self){
}


static PyObject *EventDetector_process(EventDetector* self, PyObject *args, PyObject *kwds) {
    int threshold, hdt = 1000, dead = 1000;
    PyObject *data = NULL;
    
    static char *kwlist[] = {"data", "threshold", "hdt", "dead", NULL};
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi|ii", kwlist, &data, &threshold, &hdt, &dead))
        return NULL;
    
    if (!data || !PyArray_Check(data) || PyArray_TYPE(data) != PyArray_SHORT) {
        PyErr_SetString(PyExc_TypeError, "data must be array with 'i2' dtype");
        return NULL;
    }
    
    PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew(data); 
    if (!iter)
        return NULL;
    
    while (PyArray_ITER_NOTDONE(iter)) {
        const short d = *(short*)PyArray_ITER_DATA(iter);
        const int i = iter->index;
        if (d > threshold) {
            if (self->last == -1) {
                self->last = self->event_start = i;

            } else if (i > self->last+hdt+dead) {
                if (self->callback)
                    PyObject_CallFunction(self->callback, "(ii)", self->event_start, self->last);

                self->last = self->event_start = i;

            } else if (i < self->last+hdt) {
                self->last = i;
            }
        }
        PyArray_ITER_NEXT(iter);
    }

    if (self->last != -1){
        if (self->callback)
            PyObject_CallFunction(self->callback, "(ii)", self->event_start, self->last); 
    }

    Py_RETURN_NONE;
}

static PyMethodDef EventDetector_methods[] = {
    {"process", (PyCFunction)EventDetector_process,  METH_VARARGS|METH_KEYWORDS, "Process more data"},
    {NULL}  /* Sentinel */
};

static PyTypeObject EventDetectorType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "event_detector.EventDetector",             /*tp_name*/
    sizeof(EventDetector), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)EventDetector_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    "EventDetector type",      /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    EventDetector_methods,     /* tp_methods */
    EventDetector_members,     /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)EventDetector_init,      /* tp_init */
    0,                         /* tp_alloc */
    EventDetector_new,         /* tp_new */

};

static PyMethodDef event_detector_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initevent_detector(void) {
    PyObject* m;
    if (PyType_Ready(&EventDetectorType) < 0)
        return;

    m = Py_InitModule3("event_detector", event_detector_methods, "event_detector module");
    if (m == NULL)
        return;

    import_array();

    Py_INCREF(&EventDetectorType);
    PyModule_AddObject(m, "EventDetector", (PyObject *)&EventDetectorType);
}
