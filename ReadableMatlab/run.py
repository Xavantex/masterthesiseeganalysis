import matlab.engine

eng = matlab.engine.start_matlab()
eng.Run(nargout=0)
eng.quit()