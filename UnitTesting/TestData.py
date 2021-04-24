import sys
import argparse
import scipy.io as sio
import numpy as np
import pickle

def crosscmp(dir1, dir2, file1, file2, classen, ext1 = '.pkl', ext2 = '.pkl', epoch = 'study', wfile = False,
            out = 'Datacomp.txt',
            subject = np.array(['Subj01','Subj02', 'Subj03',
                                'Subj04', 'Subj05', 'Subj06',
                                'Subj07', 'Subj08', 'Subj09',
                                'Subj11', 'Subj12', 'Subj13',
                                'Subj14', 'Subj15', 'Subj16',
                                'Subj17', 'Subj18', 'Subj19'])):

    def attri(attrim, attrip, index = None):
        def attrib():
            #if not (attrim == 'feature'):
            flag = False
            try:
                if not isinstance(attrim, str):
                    for k in range(max(len(attrim), len(attrip))):
                        try:
                            #if not attrim[k] == attrip[k]:
                            attri(attrim = attrim[k], attrip = attrip[k], index = k)
                        except IndexError:
                            print(f'Out of bounds, at index {k},\n length of mat: {len(attrim)},\n length of pickle: {len(attrip)}')
                            print(f'Mat: {attrim[k:]}, Pyth: {attrip[k:]}')

                else:
                    flag = True
                    raise TypeError
                if not wfile and flag:
                    input('Press enter to continue....')
                    flag = False
                
            except TypeError:
                if not np.array_equal(attrim, attrip):
                    if not index == None:
                        print(f'Matlab: {attrim}, python: {attrip}, index: {index}')
                        
                    else:
                        print(f'Matlab: {attrim}, python: {attrip}')
                        if not wfile:
                            input('Press enter to continue....')

            #    input('Press enter to continue....')
            #elif x == 'likelihood':
            #    print(attrim)
            #    print(attrip)

        try:
            print(f'Matlab keys: {attrim.keys()},')
            print(f'Python keys: {attrip.keys()}')
            for key in attrim:
                if key == 'feature':
                    #elif attrim == 'feature':
                        print(f'features not in the other array: ')
                        for l in np.setxor1d(attrim[key], attrip[key]):
                            print(l)
                else:
                    print(f'{key}:')
                    attri(attrim = attrim[key], attrip = attrip[key])
        except AttributeError:
            attrib()
        except SyntaxError:
            attrib()
            

        #except IndexError:
        #    print(f'Out of bounds, at index {k},\n length of mat: {len(attrim)},\n length of pickle: {len(attrip)}')

        #except TypeError as err:
        #    print(TypeError, err)
        #except AttributeError as err:
        #    print(AttributeError, err)

    def normal():
        def ifunc(sub):
            print(sub)
            if ext1 == '.pkl':
                with open(f'{dir1}/{sub}/{where}/{sub}_{epoch}_{classen}{ext1}', 'rb') as file:
                    mat = pickle.load(file)[classen]
            elif ext1 == '.mat':
                mat = sio.loadmat(f'{dir1}/{sub}/{where}/{sub}_{epoch}_{classen}{ext1}', chars_as_strings = True, simplify_cells = True)[f'{sub}_{epoch}_{classen}'] #add two options for not having loadmat
            print(f'loaded {dir1}/{sub}/{where}/{sub}_{epoch}_{classen}{ext1}')
            
            if ext2 == '.pkl':
                with open(f'{dir2}/{sub}/{where}/{sub}_{epoch}_{classen}{ext2}', 'rb') as file:
                    pik = pickle.load(file)[classen]
            elif ext2 == '.mat':
                pik = sio.loadmat(f'{dir2}/{sub}/{where}/{sub}_{epoch}_{classen}{ext2}', chars_as_strings = True, simplify_cells = True)[f'{sub}_{epoch}_{classen}']
            print(f'loaded {dir2}/{sub}/{where}/{sub}_{epoch}_{classen}{ext2}')
            attri(attrim = mat, attrip = pik)
    

        list(map(ifunc, subject))

    def randD():
        print('Random Distribution')
        #TEMPORARY
        ext1 = ext2 = '.mat'
        if ext1 == '.pkl':
            with open(f'{dir1}/{subject[-1]}/{where}/{classen}{ext1}', 'rb') as file:
                mat = pickle.load(file)[classen]
        elif ext1 == '.mat':
            mat = sio.loadmat(f'{dir1}/{subject[-1]}/{where}/{classen}{ext1}', chars_as_strings = True, simplify_cells = True)[classen] #add two options for not having loadmat
        print(f'loaded {dir1}/{subject[-1]}/{where}/{classen}{ext1}')
        
        if ext2 == '.pkl':
            with open(f'{dir2}/{subject[-1]}/{where}/{classen}{ext2}', 'rb') as file:
                pik = pickle.load(file)[classen]
        elif ext2 == '.mat':
            pik = sio.loadmat(f'{dir2}/{subject[-1]}/{where}/{classen}{ext2}', chars_as_strings = True, simplify_cells = True)[classen]
        print(f'loaded {dir2}/{subject[-1]}/{where}/{classen}{ext2}')
        
        print('loaded pyfile')
        attri(attrim = mat, attrip = pik)


    def filetest():
        if ext1 == '.pkl':
            with open(file1, 'rb') as file:
                mat = pickle.load(file)
        elif ext1 == '.mat':
            mat = sio.loadmat(file1, chars_as_strings = True, simplify_cells = True) #add two options for not having loadmat
        print(f'loaded {file1}')
        
        if ext2 == '.pkl':
            with open(file2, 'rb') as file:
                pik = pickle.load(file)
        elif ext2 == '.mat':
            pik = sio.loadmat(file2, chars_as_strings = True, simplify_cells = True)
        print(f'loaded {file2}')
        
        print('loaded pyfile')
        attri(attrim = mat, attrip = pik)


    if classen == None:
        classen = 'crossvalclass'

    switch2 = {
        'crossvalclass': '7-ClassifierTraining',
        'datapart': '7-ClassifierTraining',
        'crossvalclass_performance': '7-ClassifierTraining',
        'Random_Distribution': '8-ClassifierTesting/PermutationStudyDecodeTestVisual'
    }

    where = switch2[classen]
    
    switch = {
        'crossvalclass': normal,
        'datapart': normal,
        'crossvalclass_performance': normal,
        'Random_Distribution': randD
    }

    wfunc = switch[classen] 

    if file1 and file2:
        wfunc = filetest

    def general():

       wfunc()


    if wfile:
        orig_stdout = sys.stdout
        f = open(out, 'w')
        sys.stdout = f

    if dir1 == None:
        dir1 = './Visual'
            
    if dir2 == None:
        dir2 = './data/Visual'

    general()

    if wfile:
        sys.stdout = orig_stdout
        f.close()

if __name__ == "__main__":
    
        # Initiate the parser
    parser = argparse.ArgumentParser()

    # Add long and short argument
    parser.add_argument("--dir1", "-d1", help="set first folder path")
    parser.add_argument("--dir2", "-d2", help="set second folder path")
    parser.add_argument("--mat1", "-m1", help="pickle or mat", action="store_true")
    parser.add_argument("--mat2", "-m2", help="pickle or mat", action="store_true")
    parser.add_argument("--datapart", "-dp", help="Datapartition", action="store_true")
    parser.add_argument("--randomDist", "-rd", help="Random Distribution", action="store_true")
    parser.add_argument("--performance", "-p", help="Performance", action="store_true")
    parser.add_argument("--file1", "-f1", help="set second folder path")
    parser.add_argument("--file2", "-f2", help="set second folder path")


    # Read arguments from the command line
    args = parser.parse_args()

    pm = '.pkl'

    pm2 = '.pkl'

    dir1 = None

    dir2 = None

    purt = None

    file1 = None

    file2 = None

    # Check for --dir1
    if args.dir1:
        dir1 = args.dir1 + 'Visual'
    
    if args.dir2:
        dir2 = args.dir2 + 'Visual'

    if args.mat1:
        pm = '.mat'

    if args.mat2:
        pm2 = '.mat'
    
    if args.datapart:
        purt = 'datapart'
    
    if args.randomDist:
        purt = 'Random_Distribution'

    if args.performance:
        purt = 'crossvalclass_performance'

    if args.file1:
        file1 = args.file1
    
    if args.file2:
        file2 = args.file2

    crosscmp(dir1 = dir1, dir2 = dir2, file1 = file1, file2 = file2, ext1 = pm, ext2 = pm2, classen = purt)