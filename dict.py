'''
demos about built-in type: dict
because it's very frequently used in data analysis & machine-learning
so I think it's necessary to fully-understand it
'''

class Counter(dict):
    def __missing__(self, key):
        return 0

def printIter(x):
    try:
        while True:
            v = next(x)
            print(v)
    except StopIteration:
        print('Stop Iteration')

# constructing dicts
a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three': 3, 'one': 1, 'two': 2})
print('a: ', a) # dict is unordered
print('a == b == c == d == e: ', a == b == c == d == e)

# return the number of items in dict
print('len(a): ', len(a))

# return the item of dict with the given key, raises a KeyError if key is not in the map
print('a[\'one\']: ', a['one'])
try:
    x = a['four']
except KeyError:
    print('KeyError caught.')

# If a subclass of dict defines a method __missing__() and key is not present
# the d[key] operation calls that method with the key as argument
# then it returns or raises whatever is returned or raised by the __missing__(key) call
counter = Counter()
print('counter[0]: ', counter[0])

# change value to the given key, it's not strongly-typed
b['one'] = 'one'
print(b)

print('\'one\' in c: ', 'one' in c) # if c has a key: 'one'
del c['one'] # del item in c, both key and value
print(c)
print('\'one\' not in c: ', 'one' not in c)

x = iter(d) # equal to iter(d.keys())
printIter(x)

e.clear() # remove all items in dict
print('e: ', e)

f = d.copy() # a shallow copy
f['one'] = 4
print('d: ', d)
print('f: ', f)

# return the value for key if key is in the dictionary, else default.
# if default is not given, it defaults to None
x = a.get('four')
print('x: ', x)
x = a.get('four', 0)
print('get x with default value: ', x)

# keys() and items() return a VIEW of dict keys and items
# keys() return a set-like object, support set operations
keys = a.keys()
values = a.values()
print('a.keys(): ', list(keys))
print('a.values(): ', list(values))
del a['two']
print('a.keys(): ', list(keys))

# set operations
print(keys & {'three', 'five', 'sss'})
print(keys ^ {'another'})

# if key is in the dictionary, remove it and return its values, else return default
# if default is not given and key is not in the dictionary, a KeyError is raised
print('a.pop(\'one\'): ', a.pop('one'))
print('a.pop(\'two\', 0): ', a.pop('two', 0))
try:
    print('a.pop(\'two\'): ', a.pop('two'))
except KeyError:
    print('KeyError caught')

# if key is in the dictionary, return its value
# if not, insert key with a value of default and return default, default defaults to NOne
d.setdefault('four')
print('d: ', d)
d.setdefault('five', 0)
print('d: ', d)

# update the dictionary with the key/value pairs from other, overwriting existing keys. Return None
d.update(four=4, five=5)
print('d: ', d)

