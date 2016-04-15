class ShapeException(Exception):
    pass

import math


def shape(vec):
    try:
        len(vec[0]) > 1
        shape_expectation = (len(vec[0]),)
        equal_vectors = [shape(r) for r in vec if shape_expectation == shape(r)]
        if len(equal_vectors) != len(vec):
            raise ShapeException('Shape rule: the vectors must be the same size.')
        return (len(vec), len(vec[0]))
    except:
        return (len(vec),)


def vector_add(vec1, vec2):
    if shape(vec1)[0] == shape(vec2)[0]:
        ret = [vec1[n] + vec2[n] for n in range(shape(vec1)[0])]
    else:
        raise ShapeException('Shape rule: the vectors must be the same size.')
    return ret


def vector_sub(vec1, vec2):
    if shape(vec1)[0] == shape(vec2)[0]:
        ret = [vec1[n] - vec2[n] for n in range(shape(vec1)[0])]
    else:
        raise ShapeException('Shape rule: the vectors must be the same size.')
    return ret


def vector_sum(*vectors):
    shape_expectation = shape(vectors[0])
    equal_vectors = [shape(v) for v in vectors if shape_expectation == shape(v)]
    if len(equal_vectors) != len(vectors):
        raise ShapeException('Shape rule: the vectors must be the same size.')
    ret = []
    ret = [sum(c) for c in zip(*vectors)]
    return ret


def dot(vec1, vec2):
    if shape(vec1)[0] == shape(vec2)[0]:
        ret = [vec1[n] * vec2[n] for n in range(shape(vec1)[0])]
        ret = sum(ret)
    else:
        raise ShapeException('Shape rule: the vectors must be the same size.')
    return ret


def vector_multiply(vec, scalar):
    return [scalar*e for e in vec]


def vector_mean(*vectors):
    shape_expectation = shape(vectors[0])
    equal_vectors = [shape(v) for v in vectors if shape_expectation == shape(v)]
    if len(equal_vectors) != len(vectors):
        raise ShapeException('Shape rule: the vectors must be the same size.')
    columns = zip(*vectors)
    ret = vector_sum(*vectors)
    ret = vector_multiply(ret, 1/len(vectors))
    return ret


def magnitude(vec):
    return math.sqrt(dot(vec, vec))


def matrix_row(vec, row):
    return vec[row]


def matrix_col(vec, col):
    ret = [vec[n][col] for n, _ in enumerate(vec)]
    return ret


def matrix_scalar_multiply(vec, scalar):
    return [vector_multiply(vec[n],scalar) for n, _ in enumerate(vec)]


def matrix_vector_multiply(vec, matrix):
    if shape(vec[0]) != shape(matrix):
        raise ShapeException('Shape rule: the vectors must be the same size.')
    else:
        intermediate1 = [[row[i] for row in vec] for i in range(len(vec[0]))]
        intermediate2 = [vector_multiply(intermediate1[n], matrix[n]) for n in range(len(matrix))]
        ret = vector_sum(*intermediate2)
        return ret
























def bottom_of_file():
    pass
