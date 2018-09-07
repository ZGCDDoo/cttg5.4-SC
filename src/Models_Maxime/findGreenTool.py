#!/usr/bin/python

# Copyright Maxime Charlebois.

from numpy import *
from scipy import linalg as LA
from copy import deepcopy
import sys
import os

full_path = os.path.realpath(__file__)
pythonPathCode, file1 = os.path.split(full_path)
sys.path.insert(0, pythonPathCode + '/lib')


def main():
    fileName, verbose = inputParameter(sys.argv[:])
    set_printoptions(precision=5)

    model = ReadInputModelSimple(fileName)
    n_sites = len(model['sites'].split('\n'))

    findGreenTool(model, verbose)

    exit()

###########################################################################################################
#####################################  END OF THE PROGRAM - ###############################################
###########################################################################################################


def charHexa(digit):
    # any cluster should not have above 36 sites, even in 2050 xD
    characters = '0123456789abcdefghijklmnopqrstuvwxyz'
    if digit < 36 and digit >= 0:
        char = characters[digit]
    else:
        char = '.'
    return char


class SymmetryRule():
    def __init__(self, symmetry,  factors):
        self.symmetry = symmetry
        self.factors = factors


def findGreenTool(model, verbose=2):
    n_sites = len(model['sites'].split('\n'))

    nSitesSufficient = False

    symmetryRuleList = []

    if model['green_function_symmetries'] is not '':
        for sym in model['green_function_symmetries'].split('\n'):
            if sym.find('#') is not -1:
                sym = sym[0:sym.find('#')]
            symDef = (sym.split(':'))[0]
            symDetail = (sym.split(':'))[1].replace(' ', '')
            assert (len(symDef.split()) == n_sites) | (
                len(symDef.split()) == 2 * n_sites), ':)'
            if len(symDef.split()) == 2 * n_sites:
                nSitesSufficient = False
            symmetryRuleList.append(SymmetryRule(
                array(symDef.split()), symDetail))

    if nSitesSufficient:
        gL = n_sites
    else:
        gL = n_sites * 2

    gL = n_sites

    def swapCouple(couple):
        return (couple[1], couple[0])

    greenFunctionBasic = {}
    greenFunction = {}
    for ii in range(0, gL):
        for jj in range(0, gL):
            greenFunction[ii, jj] = symbolVar()
            greenFunctionBasic[ii, jj] = symbolVar()
            greenFunctionBasic[ii, jj].plus(
                '' + charHexa(jj) + '' + charHexa(ii) + ' ', 1.0)
            couple = (charHexa(ii), charHexa(jj))
            if ii > jj:
                couple = swapCouple(couple)
            greenFunction[ii, jj].plus(
                '' + str(couple[0]) + '' + str(couple[1]) + ' ', 1.0)

    if verbose > 0:
        print '\n\nIn the sites spaces, the most general cluster Green matrix is given by:'
        print '\ng(iw_n) ='
        print printGreenFunctionToString(greenFunctionBasic, gL)

        print '\nwhere, indices 1 to 4 represent the sites as ordered in the model for spin-up,'
        print 'and indices 5 to 8 represent the the sites for spin-down in the Nambu representation.'
        print 'Note that we always work in Nambu space. It can be used to model system without\nantiferromagnetism too.\n'

        print 'In this code, we are limited to real imaginary time Green function, hence:'
        # and G_{i,j}(iw_n) = G_{i,j}(-iw_n)'
        print 'g_{i,j}(tau) = g_{j,i}(tau) and g_{i,j}(iw_n) = g_{j,i}(iw_n)'

        print '\ng(iw_n) ='
        print printGreenFunctionToString(greenFunction, gL)

    changed = True
    while changed:
        [changed, symmetryRuleList2] = symmetrizeElementOneByOne(
            symmetryRuleList, greenFunction, gL, n_sites, verbose)

    if verbose > 0:
        print '\nIf we apply every symmetries defined by green_function_symmetries'
        print 'on this last Green function, we obtain:'
        print '\ng(iw_n) ='
        print printGreenFunctionToString(greenFunction, gL)

    listOfIndepGreenFound = {}
    N = 0
    for ii in range(0, gL):
        for jj in range(0, gL):
            if greenFunction[ii, jj].string() not in listOfIndepGreenFound:
                N += 1
                # print ii,jj
                listOfIndepGreenFound[greenFunction[ii, jj].string()] = (
                    ii, jj)

    # print(N)

    return


def symmetrizeElementOneByOne(symmetryRuleList, greenFunction, gL, n_sites, verbose):
    zero = symbolVar()
    zero.plus('0', 1.0)
    symmetryRuleList2 = []
    for element in symmetryRuleList:
        sym = []
        for ii in range(0, len(element.symmetry)):
            sym.append(int(element.symmetry[ii]))
        symmetryRuleList2.append(sym)

        for i in range(0, gL):
            for j in range(0, gL):
                permutation_i = i
                permutation_j = j
                nn = 0
                isFirst = True

                # while didn't loop on orbit
                while ((permutation_i != i) or (permutation_j != j)) or isFirst:
                    nn += 1
                    permutation_i = sym[permutation_i]
                    permutation_j = sym[permutation_j]
                    factor = 1.0
                    isConj = False

                    if (greenFunction[j, i].string() is not '0') and (greenFunction[permutation_j, permutation_i].string() is not '0'):
                        if not(greenFunction[permutation_j, permutation_i].isEqual(greenFunction[j, i], factor, isConj)):
                            if verbose > 7:
                                print element.factors, i, j, permutation_i, permutation_j, factor, greenFunction[j, i].string(), greenFunction[permutation_j, permutation_i].string()

                            name0 = (greenFunction[j, i].dic.keys())[0]
                            name1 = (
                                greenFunction[permutation_j, permutation_i].dic.keys())[0]

                            # this portion of code ensure that always the lowest name (in alphabetical order) is kept
                            if name0 < name1:
                                greenFunction[permutation_j, permutation_i].equal(
                                    greenFunction[j, i])
                                greenFunction[permutation_j,
                                              permutation_i].times(factor)
                                if isConj:
                                    greenFunction[permutation_j,
                                                  permutation_i].conj()
                            else:
                                greenFunction[j, i].equal(
                                    greenFunction[permutation_j, permutation_i])
                                greenFunction[j, i].times(factor)
                                if isConj:
                                    greenFunction[j, i].conj()
                            ####

                            if (name0 == name1):
                                greenFunction[j, i] = zero
                                greenFunction[permutation_j,
                                              permutation_i] = zero
                            if verbose > 7:
                                print element.factors, i, j, permutation_i, permutation_j, factor, greenFunction[j, i].string(), greenFunction[permutation_j, permutation_i].string()
                                print ' '

                            return [True, symmetryRuleList2]
                    isFirst = False
                    if nn > 100:
                        print 'error, too many iteration'
                        exit()
    return [False, symmetryRuleList2]


def inputParameter(argv):
    verbose = -1
    assert len(argv) > 1, 'Must input a *.model file name.'
    if '-verbose' in argv:
        position = argv.index('-verbose')
        assert position > 1, 'Must input a *.model file name.'
        verbose = int(argv[position + 1])
    return argv[1], verbose


def ReadInputModelSimple(modelFileStr):

    modelFile = ReadFile(modelFileStr)
    modelDictionnary = {}
    modelDictionnary['sites'] = readParameters('sites', modelFile)
    if findFlag('green_function_symmetries', modelFile):
        modelDictionnary['green_function_symmetries'] = readParameters(
            'green_function_symmetries', modelFile)

    return modelDictionnary


class symbolVar:
    def __init__(self):
        self.dic = {}
        self.tol = 0.0001

    def plus(self, name, coeff):
        if name[0] == '-':
            name = name[1:]
            coeff = -coeff
        if name[0] == '+':
            name = name[1:]
        if name in self.dic:
            self.dic[name] += coeff
        else:
            self.dic[name] = coeff

    def conj(self):
        tmpDic = deepcopy(self.dic)
        self.dic = {}
        for name in tmpDic:
            if name[-1] == '*':
                name2 = name[0:-1]
            else:
                name2 = name + '*'
            self.dic[name2] = tmpDic[name]

    def plusSymbol(self, symbolVarObject, factor):
        for name in symbolVarObject.dic:
            if name in self.dic:
                self.dic[name] += factor * symbolVarObject.dic[name]
            else:
                self.dic[name] = factor * symbolVarObject.dic[name]

    def equal(self, symbolVarObject):
        self.dic = {}
        for name in symbolVarObject.dic:
            self.dic[name] = deepcopy(symbolVarObject.dic[name])

    def isEqual(self, symbolVarObject, factor, isConj=False):
        isEqualBool = True
        tmpSelf = deepcopy(self)
        if isConj:
            tmpSelf.conj()
        for name in symbolVarObject.dic:
            if name in tmpSelf.dic:
                if abs(tmpSelf.dic[name] - factor * symbolVarObject.dic[name]) > self.tol:
                    isEqualBool = False
            else:
                isEqualBool = False
        for name in tmpSelf.dic:
            if name in symbolVarObject.dic:
                if abs(tmpSelf.dic[name] - factor * symbolVarObject.dic[name]) > self.tol:
                    isEqualBool = False
            else:
                isEqualBool = False
        return isEqualBool

    def times(self, value):
        for name in self.dic:
            self.dic[name] *= value

    def filter(self):
        keysToRemove = []
        for keys in self.dic:
            self.dic[keys] = round(self.dic[keys] * 1000.0) / 1000.0
            if (abs(self.dic[keys]) <= self.tol) or (keys == '0'):
                keysToRemove.append(keys)
        for keys in keysToRemove:
            self.dic.pop(keys)
        return self

    def string(self):
        output = ''
        self.filter()
        for keys in self.dic:
            if abs(self.dic[keys] - 1) <= self.tol:
                output += '+' + keys
            elif abs(self.dic[keys] + 1) <= self.tol:
                output += '-' + keys
            else:
                if self.dic[keys] > 0:
                    output += '+'
                output += '%1.2f' % self.dic[keys] + '*' + keys
        if len(output) > 0:
            if output[0] == '+':
                output = output[1:]
        else:
            output = '0'
        return output

    def isProportionalAndOrConj(self, var2):
        var3 = deepcopy(var2)
        var3.conj()

        ratio = 1
        self.filter()
        if self.dic == {}:
            if var2.dic == {}:
                return True, 1.0, False
            else:
                return False, 1.0, False
        else:
            firstKey = self.dic.keys()[0]
            val1 = self.dic[firstKey]
            if firstKey in var2.dic:
                val2 = var2.dic[firstKey]
                ratio = float(val1) / float(val2)
                tmp = deepcopy(var2)
                tmp.times(ratio)
                tmp.filter()
                if tmp.dic == self.dic:
                    return True, ratio, False
                else:
                    return False, 1.0, False

            elif firstKey in var3.dic:
                val3 = var3.dic[firstKey]
                ratio = float(val1) / float(val3)
                tmp = deepcopy(var3)
                tmp.times(ratio)
                tmp.filter()
                if tmp.dic == self.dic:
                    return True, ratio, True
                else:
                    return False, 1.0, False

            else:
                return False, 1.0, False


def unitaryTranformSparseSymbolic2(greenFunction, U, gL, verbose):

    zero = symbolVar()
    zero.plus('0', 1.0)

    Ut = conjugate(transpose(U))

    greenFunctionSym = {}
    for ii in range(0, gL):
        for jj in range(0, gL):
            greenFunctionSym[ii, jj] = symbolVar()

    for ii in range(0, gL):
        for jj in range(0, gL):
            for kk in range(0, gL):
                for ll in range(0, gL):
                    if greenFunction[ll, kk].string != '0':
                        greenFunctionSym[ii, jj].plusSymbol(
                            greenFunction[ll, kk], Ut[ii, kk] * U[ll, jj])

    listOfIndepGreenFound = {}

    for keys in greenFunctionSym:
        greenFunctionSym[keys].filter()

    for jj in range(0, gL):
        for ii in range(0, gL):
            if greenFunctionSym[0, 0].string() != '0':
                listOfIndepGreenFound['G1'] = greenFunctionSym[(0, 0)]
                break
    if (ii == jj) and (ii == gL - 1):
        print 'ERROR: No element of the matrix is non-zero'
        exit()

    if verbose > 0:
        print printGreenFunctionToString(greenFunctionSym, gL)
    greenFunctionSymRename = {}

    for jj in range(0, gL):
        for ii in range(0, gL):
            isProp = False
            tmpVar = symbolVar()
            greenFunctionSymRename[ii, jj] = zero
            for keys in listOfIndepGreenFound:
                var = listOfIndepGreenFound[keys]
                isProp, coeff, isConj = greenFunctionSym[ii, jj].isProportionalAndOrConj(
                    var)
                if isConj:
                    tmpVar.plus(keys, 1)
                    tmpVar.times(coeff)
                    tmpVar.conj()
                    break
                elif isProp:
                    tmpVar.plus(keys, 1)
                    tmpVar.times(coeff)
                    break
            if not isProp:
                if greenFunctionSym[ii, jj].dic != {}:
                    tmpVar.plus('G' + str(len(listOfIndepGreenFound) + 1), 1)
                    listOfIndepGreenFound['G' + str(
                        len(listOfIndepGreenFound) + 1)] = greenFunctionSym[ii, jj]
            greenFunctionSymRename[ii, jj] = tmpVar
    return greenFunctionSymRename, listOfIndepGreenFound, greenFunctionSym


def printGreenFunctionToString(greenFunction, gL):
    maxLen = 1
    s1 = ''
    for jj in range(0, gL):
        for ii in range(0, gL):
            maxLen = max(maxLen, len(greenFunction[ii, jj].string()))

    for jj in range(0, gL):
        for ii in range(0, gL):
            var = 1
            if (greenFunction[ii, jj].string())[0] != '-':
                s1 += ' '
                var = 0
            s1 += greenFunction[ii, jj].string()
            for kk in range(0, maxLen - len(greenFunction[ii, jj].string()) + var):
                s1 += ' '
        if jj is not gL - 1:
            s1 += '\n'
    return s1


def ReadFile(fileName):
    f = open(fileName, 'r')
    file1 = f.readlines()
    f.close

    file2 = []
    for lines in file1:
        file2.append(lines.strip())
    return file2


def readParameters(parameterName, fileList, isEssential=True):
    parameterString = ''
    if isEssential:
        assert (parameterName in fileList), 'keyword "%s" not found' % parameterName
    if parameterName in fileList:
        index1 = fileList.index(parameterName)
        for index in range(index1 + 1, len(fileList)):
            # while (line != '*') and (line != ''):
            line = fileList[index]
            if (line != '*') and (line != ''):
                if line[0] != '#':
                    if parameterString != '':
                        parameterString += '\n'
                    parameterString += line
            else:
                break
    return parameterString


def findFlag(parameterName, fileList):
    parameterFound = False
    for index in range(0, len(fileList)):
        line = fileList[index]
        lineList = line.split()
        if lineList != []:
            if lineList[0] == parameterName:
                parameterFound = True
                break
    return parameterFound


###########################################################################################################

if __name__ == "__main__":
    main()
