import neo
class spike2SimpleReader():

    def __init__(self,fileName):
        self.fileName      = fileName
        self.eventList     = list()
        self.analogSigList = list()
        self.neoReader     = False
        self.dataBlock     = False
        self.outPutData    = list()
    
    def readByNeo(self):
        self.neoReader = neo.io.Spike2IO(filename=self.fileName)
        self.dataBlock = self.neoReader.read(lazy=False)[0]

    def readSegments(self):
        self.outPutData = list()
        for seg in self.dataBlock.segments:
            self.outPutData.append(self.readSingleSeg(seg))
        

    def readSingleSeg(self,seg):
        events     = self.readEvents(seg)
        analogSigs = self.readAnalogSignals(seg)
        return (analogSigs,events)
    
    def readEvents(self,seg):
        eventData = dict()
        for event in seg.events:
            eventData[event.name] = event.times
        return eventData
  

    def readAnalogSignals(self,seg):
        analogData = dict()
        analogData['time_s']  = seg.analogsignals[0].times
        for aSignal in seg.analogsignals:
            analogData[aSignal.name] = aSignal.magnitude
        return analogData
    
    def main(self):
        self.readByNeo()
        self.readSegments()
