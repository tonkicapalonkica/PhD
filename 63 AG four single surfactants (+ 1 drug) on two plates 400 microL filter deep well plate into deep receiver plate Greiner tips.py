from opentrons import protocol_api #for the protocol to know all the commands etc.
import json #to import custom labware for the simulation. JSON is a format for storing and transporting data. json module in Python 
#is used to work with JSON (JavaScript Object Notation) data - allows to parse JSON strings and convert them into Python objects, and vice versa

#metadata
metadata = {
    "apiLevel": "2.16", #version of the API which is the code for the OT2 (can be found on the OT website)
    "protocol Name": "63 AG four single surfactants + 1 drug on two plates, 400 microL, filter deep well plate into deep receiver plate, Greiner tips",
    "description": "solubilisation power of four single surfactants for one drug in triplicates on two filter deep well plates (0.5 mL; "
    "two surfactant triplicates on one plate) into Axygen deep well plate (1.1 mL), final volume 400 microL (initial volume in wells 2/4/6/8/10B&E "
    "440 microL for serial dilution into wells 2/4/6/8/10F&G) with Greiner 300 microL pipette tips",
    "author": "Antonia"
} #viticaste zagrade don't have to be in a separate row, they can go straight after the text, but it is more clear if in a new row

#protocol run function - the whole def is the protocol
def run(protocol: protocol_api.ProtocolContext): #this row always has to be exactly like this, no changes!
    
    #1. Load labware - This includes any tips, plates, tubes, etc.

    #Pipette tips
    #P300
    #For simulation - #for Greiner custom tips, this is only needed for simulation, does not exist in reality, and only for Greiner tips
    #tiprack_path = r"C:/Users/anton/Desktop/PhD/TonTon/labware/greinerbioonesapphire_96_tiprack_300ul.json" 
    #P300tips1 = protocol.load_labware_from_definition(json.load(open(tiprack_path)), 11) #there is a way to define and import objects: name (P300tips) and then the rest
    #P300tips2 = protocol.load_labware_from_definition(json.load(open(tiprack_path)), 6)

    #For OT2
    P300tips1 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 11) #"xyz" is name of the file, number is deck slot
    P300tips2 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 3)
    P300tips3 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 2)

    #P20
    #same for simulation and for OT2 because it is a standard OT2 labware which exists in the library
    P20tips = protocol.load_labware("opentrons_96_tiprack_20ul", 7) #the number tells it the position on the deck

    #Stocks and labware
    reservoir1 = protocol.load_labware("usascientific_12_reservoir_22ml", 5) #you cannot tell it what liquid is where, just where to go (A1 etc, but we do this later)
    reservoir2 = protocol.load_labware("usascientific_12_reservoir_22ml", 6) #you cannot tell it what liquid is where, just where to go (A1 etc, but we do this later)
    
    #filter plates with receiver plates do not exist in OT library so we have to add them, and make different code for simulation and OT2, like with Greiner tips
    
    #for simulation
    #filterplatepath = r"C:/Users/anton/Desktop/PhD/TonTon/labware/milliporemultiscreensolvinert0.5mlfilterplates0.45hydrophilicptfewith1.1mlaxygendeepwellplatereceiver_96_wellplate_500ul.json"
    #fplate1 = protocol.load_labware_from_definition(json.load(open(filterplatepath)), 8) #fplate = filter plate with receiver plate
    #fplate2 = protocol.load_labware_from_definition(json.load(open(filterplatepath)), 9)  

    #for OT2
    fplate1 = protocol.load_labware("milliporemultiscreensolvinert0.5mlfilterplates0.45hydrophilicptfewith1.1mlaxygendeepwellplatereceiver_96_wellplate_500ul", 8)
    fplate2 = protocol.load_labware("milliporemultiscreensolvinert0.5mlfilterplates0.45hydrophilicptfewith1.1mlaxygendeepwellplatereceiver_96_wellplate_500ul", 9)
    
    #2. load Pipettes (OT calles them instrument)
    P300single = protocol.load_instrument("p300_single_gen2", "left", tip_racks=[P300tips1, P300tips2, P300tips3])
    P20single = protocol.load_instrument("p20_single_gen2", "right", tip_racks=[P20tips])

    #set blow out flow rate
    flowrate_blowout = 12 #it is not possible to set the flow rate for each step individually, just globally for each pipette. This is unnecessarily slow for water, but needed for surfactants.
    P300single.flow_rate.blow_out = flowrate_blowout
    P20single.flow_rate.blow_out = flowrate_blowout

    #P20single.set_flow_rate(aspirate=2, dispense=1.2) #function can be called at any time to set the dispense and aspirate flow rates for the pipette, in ul/s. The dispense speed also controls the speed of blow_out.?
    #P300single.set_flow_rate(aspirate=20, dispense=12) #ChatGPT says default is 92.86 uL/s for aspirate and 92.86 uL/s for dispense, OT2 website says 46.43 uL/s for both. Judging by the information of actual dispensing/aspirating speeds given by the OT2 app, ChatGPT is correct.

    #3. Steps for experiments:
    #3.a surfactants

    #plate1

    #transfer 400 uL of surfactant 1 stock 1 (A1 of reservoir1) into well A1 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A1"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A1"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A1"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 1 stock 1 (A1 of reservoir1) into well B1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A1"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 170, location = fplate1["B1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 1 stock 1 (A1 of reservoir1) into well C1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C1"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 240 uL of surfactant 1 stock 1 (A1 of reservoir1) into well D1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D1"], rate = 0.13) 
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 1 stock 1 (A1 of reservoir1) into well E1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 1 stock 1 (A1 of reservoir1) into well F1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 1 stock 1 (A1 of reservoir1) into well G1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G1"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 60 uL of surfactant 1 stock 1 (A1 of reservoir1) into well H1 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H1"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 1 stock 1 (A1 of reservoir1) into well A2 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A1"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A2"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 1 stock 1 (A1 of reservoir1) into well B2 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F2
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["B2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A1"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B2"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 1 stock 1 (A1 of reservoir1) into well C2 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate1["C2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 1 stock 1 (A1 of reservoir1) into well D2 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate1["D2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 1 stock 1 (A1 of reservoir1) into well E2 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G2
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["E2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 1 stock 2 (A2 of reservoir1) into well A3 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A2"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A3"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A2"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 1 stock 2 (A2 of reservoir1) into well B3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B3"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 1 stock 2 (A2 of reservoir1) into well C3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 1 stock 2 (A2 of reservoir1) into well D3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D3"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 1 stock 2 (A2 of reservoir1) into well E3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 1 stock 2 (A2 of reservoir1) into well F3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 1 stock 2 (A2 of reservoir1) into well G3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 1 stock 2 (A2 of reservoir1) into well H3 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 1 stock 2 (A2 of reservoir1) into well A4 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A2"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A4"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 1 stock 2 (A2 of reservoir1) into well B4 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F4
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["B4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A2"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B4"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 1 stock 2 (A2 of reservoir1) into well C4 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate1["C4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 1 stock 2 (A2 of reservoir1) into well D4 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate1["D4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 1 stock 2 (A2 of reservoir1) into well E4 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G4
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["E4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 1 stock 3 (A3 of reservoir1) into well A5 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A3"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A5"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A3"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 1 stock 3 (A3 of reservoir1) into well B5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B5"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B5"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 1 stock 3 (A3 of reservoir1) into well C5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C5"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 1 stock 3 (A3 of reservoir1) into well D5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D5"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 1 stock 3 (A3 of reservoir1) into well E5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 1 stock 3 (A3 of reservoir1) into well F5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 1 stock 3 (A3 of reservoir1) into well G5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 1 stock 3 (A3 of reservoir1) into well H5 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 1 stock 3 (A3 of reservoir1) into well A6 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A3"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A6"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 1 stock 3 (A3 of reservoir1) into well B6 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F6
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["B6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A3"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 1 stock 3 (A3 of reservoir1) into well C6 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 18, location = fplate1["C6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 1 stock 3 (A3 of reservoir1) into well D6 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 14, location = fplate1["D6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 1 stock 3 (A3 of reservoir1) into well E6 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G6
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["E6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 2 stock 1 (A6 of reservoir1) into well A7 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A6"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A7"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A6"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 2 stock 1 (A6 of reservoir1) into well B7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A6"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 170, location = fplate1["B7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 2 stock 1 (A6 of reservoir1) into well C7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C7"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 240 uL of surfactant 2 stock 1 (A6 of reservoir1) into well D7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D7"], rate = 0.13) 
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 2 stock 1 (A6 of reservoir1) into well E7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 2 stock 1 (A6 of reservoir1) into well F7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 2 stock 1 (A6 of reservoir1) into well G7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G7"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 60 uL of surfactant 2 stock 1 (A6 of reservoir1) into well H7 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H7"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 2 stock 1 (A6 of reservoir1) into well A8 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A6"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A8"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 2 stock 1 (A6 of reservoir1) into well B8 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F8
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["B8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A6"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B8"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 2 stock 1 (A6 of reservoir1) into well C8 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate1["C8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 2 stock 1 (A6 of reservoir1) into well D8 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate1["D8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 2 stock 1 (A6 of reservoir1) into well E8 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G8
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["E8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 2 stock 2 (A7 of reservoir1) into well A9 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A7"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A9"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A7"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 2 stock 2 (A7 of reservoir1) into well B9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B9"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 2 stock 2 (A7 of reservoir1) into well C9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 2 stock 2 (A7 of reservoir1) into well D9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D9"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 2 stock 2 (A7 of reservoir1) into well E9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 2 stock 2 (A7 of reservoir1) into well F9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 2 stock 2 (A7 of reservoir1) into well G9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 2 stock 2 (A7 of reservoir1) into well H9 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 2 stock 2 (A7 of reservoir1) into well A10 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A7"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A10"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 2 stock 2 (A7 of reservoir1) into well B10 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F10
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["B10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A7"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B10"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 2 stock 2 (A7 of reservoir1) into well C10 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate1["C10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 2 stock 2 (A7 of reservoir1) into well D10 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate1["D10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 2 stock 2 (A7 of reservoir1) into well E10 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G10
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["E10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 2 stock 3 (A8 of reservoir1) into well A11 of first plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir1["A8"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate1["A11"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir1["A8"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate1["A11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 2 stock 3 (A8 of reservoir1) into well B11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B11"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate1["B11"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 2 stock 3 (A8 of reservoir1) into well C11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate1["C11"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 2 stock 3 (A8 of reservoir1) into well D11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate1["D11"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 2 stock 3 (A8 of reservoir1) into well E11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate1["E11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 2 stock 3 (A8 of reservoir1) into well F11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate1["F11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 2 stock 3 (A8 of reservoir1) into well G11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate1["G11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 2 stock 3 (A8 of reservoir1) into well H11 of first plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir1["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate1["H11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 2 stock 3 (A8 of reservoir1) into well A12 of first plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir1["A8"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate1["A12"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["A12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 2 stock 3 (A8 of reservoir1) into well B12 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well F12
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["B12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir1["A8"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate1["B12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 2 stock 3 (A8 of reservoir1) into well C12 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir1["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 18, location = fplate1["C12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 2 stock 3 (A8 of reservoir1) into well D12 of first plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir1["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 14, location = fplate1["D12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 2 stock 3 (A8 of reservoir1) into well E12 of first plate *V = 440 microL to enable a 1 in 10 serial dilution into well G12
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir1["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate1["E12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()    


    #plate2

    #transfer 400 uL of surfactant 3 stock 1 (A1 of reservoir2) into well A1 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A1"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A1"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A1"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 3 stock 1 (A1 of reservoir2) into well B1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A1"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 170, location = fplate2["B1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 3 stock 1 (A1 of reservoir2) into well C1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C1"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 240 uL of surfactant 3 stock 1 (A1 of reservoir2) into well D1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D1"], rate = 0.13) 
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 3 stock 1 (A1 of reservoir2) into well E1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 3 stock 1 (A1 of reservoir2) into well F1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F1"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 3 stock 1 (A1 of reservoir2) into well G1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G1"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 60 uL of surfactant 3 stock 1 (A1 of reservoir2) into well H1 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A1"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H1"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 3 stock 1 (A1 of reservoir2) into well A2 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A1"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A2"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 3 stock 1 (A1 of reservoir2) into well B2 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F2
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["B2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A1"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B2"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 3 stock 1 (A1 of reservoir2) into well C2 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate2["C2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 3 stock 1 (A1 of reservoir2) into well D2 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate2["D2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 3 stock 1 (A1 of reservoir2) into well E2 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G2
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A1"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["E2"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 3 stock 2 (A2 of reservoir2) into well A3 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A2"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A3"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A2"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 3 stock 2 (A2 of reservoir2) into well B3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B3"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 3 stock 2 (A2 of reservoir2) into well C3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C3"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 3 stock 2 (A2 of reservoir2) into well D3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D3"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 3 stock 2 (A2 of reservoir2) into well E3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 3 stock 2 (A2 of reservoir2) into well F3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 3 stock 2 (A2 of reservoir2) into well G3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 3 stock 2 (A2 of reservoir2) into well H3 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A2"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H3"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 3 stock 2 (A2 of reservoir2) into well A4 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A2"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A4"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 3 stock 2 (A2 of reservoir2) into well B4 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F4
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["B4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A2"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B4"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 3 stock 2 (A2 of reservoir2) into well C4 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate2["C4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 3 stock 2 (A2 of reservoir2) into well D4 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate2["D4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 3 stock 2 (A2 of reservoir2) into well E4 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G4
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A2"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["E4"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 3 stock 3 (A3 of reservoir2) into well A5 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A3"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A5"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A3"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 3 stock 3 (A3 of reservoir2) into well B5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B5"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B5"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 3 stock 3 (A3 of reservoir2) into well C5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C5"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 3 stock 3 (A3 of reservoir2) into well D5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D5"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 3 stock 3 (A3 of reservoir2) into well E5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 3 stock 3 (A3 of reservoir2) into well F5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 3 stock 3 (A3 of reservoir2) into well G5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 3 stock 3 (A3 of reservoir2) into well H5 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A3"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H5"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 3 stock 3 (A3 of reservoir2) into well A6 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A3"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A6"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 3 stock 3 (A3 of reservoir2) into well B6 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F6
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["B6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A3"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 3 stock 3 (A3 of reservoir2) into well C6 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 18, location = fplate2["C6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 3 stock 3 (A3 of reservoir2) into well D6 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 14, location = fplate2["D6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 3 stock 3 (A3 of reservoir2) into well E6 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G6
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A3"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["E6"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 4 stock 1 (A6 of reservoir2) into well A7 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A6"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A7"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A6"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 4 stock 1 (A6 of reservoir2) into well B7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A6"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 170, location = fplate2["B7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 4 stock 1 (A6 of reservoir2) into well C7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C7"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 240 uL of surfactant 4 stock 1 (A6 of reservoir2) into well D7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D7"], rate = 0.13) 
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 4 stock 1 (A6 of reservoir2) into well E7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 4 stock 1 (A6 of reservoir2) into well F7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F7"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 4 stock 1 (A6 of reservoir2) into well G7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G7"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 60 uL of surfactant 4 stock 1 (A6 of reservoir2) into well H7 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A6"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H7"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 4 stock 1 (A6 of reservoir2) into well A8 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A6"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A8"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 4 stock 1 (A6 of reservoir2) into well B8 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F8
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["B8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A6"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B8"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 4 stock 1 (A6 of reservoir2) into well C8 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate2["C8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 4 stock 1 (A6 of reservoir2) into well D8 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate2["D8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 4 stock 1 (A6 of reservoir2) into well E8 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G8
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A6"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["E8"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 4 stock 2 (A7 of reservoir2) into well A9 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A7"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A9"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A7"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 4 stock 2 (A7 of reservoir2) into well B9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B9"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 4 stock 2 (A7 of reservoir2) into well C9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C9"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 4 stock 2 (A7 of reservoir2) into well D9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D9"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 4 stock 2 (A7 of reservoir2) into well E9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 4 stock 2 (A7 of reservoir2) into well F9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 4 stock 2 (A7 of reservoir2) into well G9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 4 stock 2 (A7 of reservoir2) into well H9 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A7"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H9"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 4 stock 2 (A7 of reservoir2) into well A10 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A7"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A10"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 4 stock 2 (A7 of reservoir2) into well B10 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F10
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["B10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A7"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B10"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 4 stock 2 (A7 of reservoir2) into well C10 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 18, location = fplate2["C10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 4 stock 2 (A7 of reservoir2) into well D10 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 14, location = fplate2["D10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 4 stock 2 (A7 of reservoir2) into well E10 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G10
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A7"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["E10"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #transfer 400 uL of surfactant 4 stock 3 (A8 of reservoir2) into well A11 of second plate
    P300single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P300single.aspirate(volume = 200, location = reservoir2["A8"].bottom(z=3), rate = 0.22) #rate is fraction of default rate. For P300 single default rate stated as 46.43 uL/s on ProtocolDesigner. Required aspiration rate 20 uL/s = 0.43 x 46.4. However, OT2 app says that with 0.43 it is aspirating at a rate of 39.9298 uL/s. Subsequently, rate has been changed to 1/2 of 0.43 = 0.22.
    P300single.dispense(volume = 200, location = fplate2["A11"], rate = 0.13) #default rate for P300 single 92.86 uL/s. required dispensing rate 12 uL/s = 0.13 x 92.8. 
    P300single.blow_out() #protocol here is OK with blow out in the dispense command, but OT2 app doesn't like it, so written out in a separate command
    P300single.touch_tip()
    P300single.aspirate(volume = 200, location = reservoir2["A8"].bottom(z=3), rate = 0.22) 
    P300single.dispense(volume = 200, location = fplate2["A11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 340 uL of surfactant 4 stock 3 (A8 of reservoir2) into well B11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 170, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B11"], rate = 0.13)
    P300single.blow_out() 
    P300single.touch_tip()
    P300single.aspirate(volume = 170, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 170, location = fplate2["B11"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 300 uL of surfactant 4 stock 3 (A8 of reservoir2) into well C11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 300, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 300, location = fplate2["C11"], rate = 0.13) 
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 240 uL of surfactant 4 stock 3 (A8 of reservoir2) into well D11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 240, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 240, location = fplate2["D11"], rate = 0.13)  
    P300single.blow_out() 
    P300single.touch_tip()

    #transfer 200 uL of surfactant 4 stock 3 (A8 of reservoir2) into well E11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 200, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 200, location = fplate2["E11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 140 uL of surfactant 4 stock 3 (A8 of reservoir2) into well F11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 140, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 140, location = fplate2["F11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 100 uL of surfactant 4 stock 3 (A8 of reservoir2) into well G11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 100, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 100, location = fplate2["G11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #transfer 60 uL of surfactant 4 stock 3 (A8 of reservoir2) into well H11 of second plate
    #no pick up tip - keeping the tip from the previous step
    P300single.aspirate(volume = 60, location = reservoir2["A8"].bottom(z=3), rate = 0.22)
    P300single.dispense(volume = 60, location = fplate2["H11"], rate = 0.13)
    P300single.blow_out()
    P300single.touch_tip()

    #dispose of pipette tip
    P300single.drop_tip()

    #transfer 22 uL of surfactant 4 stock 3 (A8 of reservoir2) into well A12 of second plate
    P20single.pick_up_tip() #bracket with nothing inside means the next available tip as per default
    P20single.aspirate(volume = 11, location = reservoir2["A8"].bottom(z=3), rate = 0.27) #rate is fraction of default rate. for P20, default = 3.78 uL/s. I want rate 2 uL/s = 0.54 x 3.7. However, OT2 app says that with 0.54 it is aspirating at a rate of 4.0824 uL/s. Subsequently, rate has been changed to 1/2 of 0.54 = 0.27.
    P20single.dispense(volume = 11, location = fplate2["A12"], rate = 0.16) #rate is fraction of default rate. for P20, default = 7.56 uL/s. I want rate 1.2 uL/s = 0.16 x 7.56
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["A12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 22 uL of surfactant 4 stock 3 (A8 of reservoir2) into well B12 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well F12
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["B12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()
    P20single.aspirate(volume = 11, location = reservoir2["A8"].bottom(z=3), rate = 0.27) 
    P20single.dispense(volume = 11, location = fplate2["B12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 18 uL of surfactant 4 stock 3 (A8 of reservoir2) into well C12 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 18, location = reservoir2["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 18, location = fplate2["C12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 14 uL of surfactant 4 stock 3 (A8 of reservoir2) into well D12 of second plate
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 14, location = reservoir2["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 14, location = fplate2["D12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #transfer 11 uL of surfactant 4 stock 3 (A8 of reservoir2) into well E12 of second plate *V = 440 microL to enable a 1 in 10 serial dilution into well G12
    #no pick up tip - keeping the tip from the previous step
    P20single.aspirate(volume = 11, location = reservoir2["A8"].bottom(z=3), rate = 0.27)
    P20single.dispense(volume = 11, location = fplate2["E12"], rate = 0.16)
    P20single.blow_out()
    P20single.touch_tip()

    #dispose of pipette tip
    P20single.drop_tip()

    #3.b HEPES

    #transfer 400 uL of HEPES (A4 of reservoir1&2) into wells 2,4,6,8,10,12H of first and second plate

    #single code for all wells of same volume
    #the code for each plate has to be written separately
    #first plate
    #a list has to be written for the destination wells
    dest_wells1_H = ["H2", "H4", "H6", "H8", "H10", "H12"]
    P300single.transfer(volume = 200, source = reservoir1["A4"].bottom(z=3), dest = [fplate1[well].bottom(z=1) for well in dest_wells1_H], 
                        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #rate not written as default 46.4 uL/s ok for water
    #new_tip = "once" means that a new tip is picked up only once for the first transfer, and the same tip is reused for all transfers.
    P300single.transfer(volume = 200, source = reservoir1["A4"].bottom(z=3), dest = [fplate1[well].bottom(z=1) for well in dest_wells1_H],
                         blow_out = True, blowout_location = "destination well", touch_tip = True)
    #new_tip = "never" means that a new tip is not picked up and the same tip is reused for all transfers, however OT2 app doesn't like it
    #so new_tip removed in second step and in the next command for second plate

    #second plate
    #a list has to be written for the destination wells
    dest_wells2_H = ["H2", "H4", "H6", "H8", "H10", "H12"]
    P300single.transfer(volume = 200, source = reservoir2["A4"].bottom(z=3), dest = [fplate2[well].bottom(z=1) for well in dest_wells2_H], 
                        blow_out = True, blowout_location = "destination well", touch_tip = True) #rate not written as default 46.4 uL/s ok for water
    P300single.transfer(volume = 200, source = reservoir2["A4"].bottom(z=3), dest = [fplate2[well].bottom(z=1) for well in dest_wells2_H], 
                        blow_out = True, blowout_location = "destination well", touch_tip = True) #rate not written as default 46.4 uL/s ok for water
    
    #remaining wells with different volumes
    #first plate
    #transfer of HEPES (A4 of reservoir1) into wells B1 - G2 of first plate (surfactant 1 replicate 1)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir1) into wells B1-G2 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A4"].bottom(z=3),
        dest = fplate1.wells("B1", "C1", "D1", "E1", "F1", "G1", "H1", "A2", "B2", "C2", "D2", "E2", "F2", "G2"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL
    
    #transfer of HEPES (A4 of reservoir1) into wells B3 - G4 of first plate (surfactant 1 replicate 2)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir1) into wells B3-G4 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A4"].bottom(z=3),
        dest = fplate1.wells("B3", "C3", "D3", "E3", "F3", "G3", "H3", "A4", "B4", "C4", "D4", "E4", "F4", "G4"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A4 of reservoir1) into wells B5 - G6 of first plate (surfactant 1 replicate 3)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir1) into wells B5-G6 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A4"].bottom(z=3),
        dest = fplate1.wells("B5", "C5", "D5", "E5", "F5", "G5", "H5", "A6", "B6", "C6", "D6", "E6", "F6", "G6"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A5 of reservoir1) into wells B7 - G8 of first plate (surfactant 2 replicate 1)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir1) into wells B7-G8 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A5"].bottom(z=3),
        dest = fplate1.wells("B7", "C7", "D7", "E7", "F7", "G7", "H7", "A8", "B8", "C8", "D8", "E8", "F8", "G8"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL
    
    #transfer of HEPES (A5 of reservoir1) into wells B9 - G10 of first plate (surfactant 2 replicate 2)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir1) into wells B9-G10 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A5"].bottom(z=3),
        dest = fplate1.wells("B9", "C9", "D9", "E9", "F9", "G9", "H9", "A10", "B10", "C10", "D10", "E10", "F10", "G10"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A5 of reservoir1) into wells B11 - G12 of first plate (surfactant 2 replicate 3)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir1) into wells B11-G12 of first plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir1["A5"].bottom(z=3),
        dest = fplate1.wells("B11", "C11", "D11", "E11", "F11", "G11", "H11", "A12", "B12", "C12", "D12", "E12", "F12", "G12"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #second plate
    #transfer of HEPES (A4 of reservoir2) into wells B1 - G2 of second plate (surfactant 3 replicate 1)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir2) into wells B1-G2 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A4"].bottom(z=3),
        dest = fplate2.wells("B1", "C1", "D1", "E1", "F1", "G1", "H1", "A2", "B2", "C2", "D2", "E2", "F2", "G2"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL
    
    #transfer of HEPES (A4 of reservoir2) into wells B3 - G4 of second plate (surfactant 3 replicate 2)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir2) into wells B3-G4 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A4"].bottom(z=3),
        dest = fplate2.wells("B3", "C3", "D3", "E3", "F3", "G3", "H3", "A4", "B4", "C4", "D4", "E4", "F4", "G4"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A4 of reservoir2) into wells B5 - G6 of second plate (surfactant 3 replicate 3)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A4 of reservoir2) into wells B5-G6 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A4"].bottom(z=3),
        dest = fplate2.wells("B5", "C5", "D5", "E5", "F5", "G5", "H5", "A6", "B6", "C6", "D6", "E6", "F6", "G6"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A5 of reservoir2) into wells B7 - G8 of second plate (surfactant 4 replicate 1)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir2) into wells B7-G8 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A5"].bottom(z=3),
        dest = fplate2.wells("B7", "C7", "D7", "E7", "F7", "G7", "H7", "A8", "B8", "C8", "D8", "E8", "F8", "G8"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL
    
    #transfer of HEPES (A5 of reservoir2) into wells B9 - G10 of second plate (surfactant 4 replicate 2)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir2) into wells B9-G10 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A5"].bottom(z=3),
        dest = fplate2.wells("B9", "C9", "D9", "E9", "F9", "G9", "H9", "A10", "B10", "C10", "D10", "E10", "F10", "G10"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

    #transfer of HEPES (A5 of reservoir2) into wells B11 - G12 of second plate (surfactant 4 replicate 3)
    #transfer of 60, 80, 100, 120, 160, 300, 340, 378, 418, 382, 386, 429, 360, 360 uL of HEPES (A5 of reservoir2) into wells B11-G12 of second plate
    P300single.transfer(
        volume = [60, 100, 160, 200, 260, 300, 340, 378, 418, 382, 386, 429, 360, 360],
        source = reservoir2["A5"].bottom(z=3),
        dest = fplate2.wells("B11", "C11", "D11", "E11", "F11", "G11", "H11", "A12", "B12", "C12", "D12", "E12", "F12", "G12"),
        mix_after = (3, 300), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #in mix_after, first number is how many times, second number is mixing volume in uL

       
    #at this point, A1,3,5,7,9,11 to E2,4,6,8,10,12 of each plate are fully done + water of wells F/G2,4,6,8,10,12 

    #the remaining wells F/G2,4,6,8,10,12 will be done below

    #3.c serial dilution from wells with 440 uL (B2,etc. and E2,etc.) to wells F/G2,etc. of each plate

    #first plate
    #transfer of 40 uL of more concentrated surfactant from wells B2, B4, B6, B8, B10 and B12 of first plate to wells F2, F4, F6, F8, F10 and F12 of first plate - all surfactants all replicates
    P20single.transfer(
        volume = 40,
        source = fplate1.wells("B2", "B4", "B6", "B8", "B10", "B12"),
        dest = fplate1.wells("F2", "F4", "F6", "F8", "F10", "F12"),
        new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True)
    
    #second plate
    #transfer of 40 uL of more concentrated surfactant from wells B2, B4, B6, B8, B10 and B12 of second plate to wells F2, F4, F6, F8, F10 and F12 of second plate - all surfactants all replicates
    P20single.transfer(
        volume = 40,
        source = fplate2.wells("B2", "B4", "B6", "B8", "B10", "B12"),
        dest = fplate2.wells("F2", "F4", "F6", "F8", "F10", "F12"),
        new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True)

    #protocol completed
    #print statement to indicate protocol completion
    protocol.comment("☀️🌼🌹")