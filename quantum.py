import numpy, sympy, sys, itertools, functools, operator

i = sympy.I

def X(size, level):
	matrix = [[0] * i + [1] + [0] * (2 ** size + ~i) for i in range(2 ** size)]
	rate = 2 ** (size - level)
	for i in range(0, 2 ** size, rate * 2):
		matrix[i:i + rate], matrix[i + rate:i + 2 * rate] = matrix[i + rate:i + 2 * rate], matrix[i:i + rate]
	return numpy.array(matrix)

def Y(size, level):
	block = 2 ** (size - level)
	return numpy.diag([1 if i // block % 2 == 0 else -1 for i in range(2 ** size)])

def Z(size, level):
	return numpy.dot(X(size, level), Y(size, level)) * i

transformations = {
	"X": X,
	"Y": Y,
	"Z": Z,
}

def combine(vectors):
	return numpy.array([functools.reduce(operator.mul, i) for i in itertools.product(*vectors)])

def process(code):
	lines = list(map(list, code.strip("\n").splitlines()))
	length = max(map(len, lines))
	lines = [line + [" "] * length for line in lines]
	code = list(map(list, zip(*lines)))
	values = combine([numpy.array([1, 0]) if char == "0" else numpy.array([0, 1]) if char == "1" else None for char in code[0]])
	for row in code[1:]:
		indices = []
		for index, char in enumerate(row):
			if char == "|":
				if indices:
					indices[-1].append(index)
			elif char != " ":
				indices.append([index])
		matrix = numpy.diag([1] * len(values))
		for indexlist in indices:
			matrix = numpy.dot(matrix, transformations[row[indexlist[0]]](len(lines), indexlist[0] + 1))
		values = numpy.dot(matrix, values)
	return values

def prettyformat(array):
	if hasattr(array, "__iter__") and not isinstance(array, str):
		return "[%s]" % ", ".join(map(prettyformat, array))
	else:
		return "%s" % array

print(prettyformat(process(sys.stdin.read())))
