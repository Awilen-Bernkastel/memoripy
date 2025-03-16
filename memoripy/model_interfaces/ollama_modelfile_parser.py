#ollama_modelfile_parser.py
# Apache 2.0 license, Created by Awilen Bernkastel

class modelfile:
    def __init__(self):
        self.ofrom = ""
        self.otemplate = ""
        self.oparameters = {}
        self.osystem = ""
        self.oadapter = ""
        self.olicense = ""
        self.ommessage = ""

    def parse(self, modelcard):
        modelcard = modelcard.modelfile.split("\n")
        current_parsed_elements = None
        for line in modelcard:
            if line.startswith("#"):
                continue
            else:
                line = line.split(" ")
                if line[0].startswith("FROM"):
                    self.ofrom = " ".join(line[1:])
                elif line[0].startswith("PARAMETER"):
                    if line[1] in ["mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "repeat_last_n",
                                   "repeat_penalty", "temperature", "seed", "stop", "num_predict", "top_k",
                                   "top_p", "num_p"]:
                        self.oparameters[line[1]] = " ".join(line[2:])
                    current_parsed_elements = None
                elif line[0].startswith("SYSTEM"):
                    self.osystem = (" ".join(line[1:])).strip("\"\"\"")
                    current_parsed_elements = "osystem"
                elif line[0].startswith("ADAPTER"):
                    self.oadapter = " ".join(line[1:])
                    current_parsed_elements = None
                elif line[0].startswith("LICENSE"):
                    self.olicense = (" ".join(line[1:])).strip("\"\"\"")
                    current_parsed_elements = "olicense"
                elif line[0].startswith("MESSAGE"):
                    self.ommessage = (" ".join(line[1:])).strip("\"\"\"")
                    current_parsed_elements = "omessage"
                else:
                    if current_parsed_elements is not None:
                        setattr(self, current_parsed_elements, getattr(self, current_parsed_elements) + "\n" + (" ".join(line)).strip("\"\"\""))
        return self
