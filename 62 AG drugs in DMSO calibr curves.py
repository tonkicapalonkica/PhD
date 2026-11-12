from opentrons import protocol_api #for the protocol to know all the commands etc.
import json #to import custom labware for the simulation. JSON is a format for storing and transporting data. json module in Python 
#is used to work with JSON (JavaScript Object Notation) data - allows to parse JSON strings and convert them into Python objects, and vice versa

#metadata
metadata = {
    "apiLevel": "2.8", #version of the API which is the code for the OT2 (can be found on the OT website)
    "protocol Name": "62 AG drugs in DMSO calibration curves, 200 microL, UV-transparent plate, Greiner tips",
    "description": "calibration curves for drugs in DMSO, one drug with triplicates per UV-transparent plate, "
    "final volume 200 microL (initial volume in wells A1-H1 200 microL, wells A2-H2 220 microL for serial dilution into wells"
    "A3-H3) with Greiner 300 microL pipette tips and OT 20 microL pipette tips",
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
    P300tips1 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 6) #"xyz" is name of the file, number is deck slot
    P300tips2 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 11)
    P300tips3 = protocol.load_labware("greinerbioonesapphire_96_tiprack_300ul", 3)

    #P20
    #same for simulation and for OT2 because it is a standard OT2 labware which exists in the library
    P20tips = protocol.load_labware("opentrons_96_tiprack_20ul", 7) #the number tells it the position on the deck

    #Stocks and labware
    reservoir = protocol.load_labware("usascientific_12_reservoir_22ml", 5) #you cannot tell it what liquid is where, just where to go (A1 etc, but we do this later)
    
    #for OT2
    plate1 = protocol.load_labware("corning_96_wellplate_360ul_flat", 8)
    plate2 = protocol.load_labware("corning_96_wellplate_360ul_flat", 9)
    
    #2. load Pipettes (OT calles them instrument)
    P300single = protocol.load_instrument("p300_single_gen2", "left", tip_racks=[P300tips1, P300tips2, P300tips3])
    P20single = protocol.load_instrument("p20_single_gen2", "right", tip_racks=[P20tips])

    #functions to set the dispense and aspirate flow rates for the pipette, in ul/s. The dispense speed also controls the speed of blow_out.?
    P20single.aspirate_flow_rate = 1
    P20single.dispense_flow_rate = 1
    P20single.blow_out_flow_rate = 1
    P300single.aspirate_flow_rate = 10
    P300single.dispense_flow_rate = 10 #ChatGPT says default is 92.86 uL/s for aspirate and 92.86 uL/s for dispense, OT2 website says 46.43 uL/s for both. Judging by the information of actual dispensing/aspirating speeds given by the OT2 app, ChatGPT is correct.
    P300single.blow_out_flow_rate = 10

    #3. Steps for experiments:

    #3.a transfer of drugs in DMSO stocks into first and second columns of each plate for all replicates

    #first columns for first plate all replicates (columns 1, 4 and 7 for first, second and third replicate, respectively)

    #transfer of drug 1 stock 1 (A1 of reservoir) into wells A1-H1 of first plate (first replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 1 stock 1 (A1 of reservoir) into wells A1-H1 of first plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A1"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #new_tip = "once" means that a new tip is picked up only once for the first transfer, and the same tip is reused for all transfers.

    #transfer of drug 1 stock 2 (A2 of reservoir) into wells A4-H4 of first plate (second replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 1 stock 2 (A2 of reservoir) into wells A4-H4 of first plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A2"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #transfer of drug 1 stock 3 (A3 of reservoir) into wells A7-H7 of first plate (third replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 1 stock 3 (A3 of reservoir) into wells A7-H7 of first plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A3"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #first columns for second plate all replicates (columns 1, 4 and 7 for first, second and third replicate, respectively)

    #transfer of drug 2 stock 1 (A6 of reservoir) into wells A1-H1 of second plate (first replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 1 stock 1 (A1 of reservoir) into wells A1-H1 of first plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A6"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #transfer of drug 2 stock 2 (A7 of reservoir) into wells A4-H4 of second plate (second replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 2 stock 2 (A7 of reservoir) into wells A4-H4 of second plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A7"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #transfer of drug 2 stock 3 (A8 of reservoir) into wells A7-H7 of second plate (third replicate)
    #transfer of 200, 180, 160, 130, 100, 80, 60 and 40 uL of drug 2 stock 3 (A8 of reservoir) into wells A7-H7 of second plate
    P300single.transfer(
        volume = [200, 180, 160, 130, 100, 80, 60, 40],
        source = reservoir["A8"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix

    #second columns (approx. 10x dilution from first column) for first plate all replicates (columns 2, 5 and 8 for first, second and third replicate, respectively)

    #transfer of drug 1 stock 1 (A1 of reservoir) into wells A2-H2 of first plate (first replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 1 stock 1 (A1 of reservoir) into wells A2-H2 of first plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A1"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A2", "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #transfer of drug 1 stock 2 (A2 of reservoir) into wells A5-H5 of first plate (second replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 1 stock 2 (A2 of reservoir) into wells A5-H5 of first plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A2"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A5", "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix

    #transfer of drug 1 stock 3 (A3 of reservoir) into wells A8-H8 of first plate (third replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 1 stock 3 (A3 of reservoir) into wells A8-H8 of first plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A3"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A8", "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #second columns (approx. 10x dilution from first column) for second plate all replicates (columns 2, 5 and 8 for first, second and third replicate, respectively)

    #transfer of drug 2 stock 1 (A6 of reservoir) into wells A2-H2 of second plate (first replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 2 stock 1 (A6 of reservoir) into wells A2-H2 of second plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A6"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A2", "A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #transfer of drug 2 stock 2 (A7 of reservoir) into wells A5-H5 of second plate (second replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 2 stock 2 (A7 of reservoir) into wells A5-H5 of second plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A7"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A5", "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix

    #transfer of drug 2 stock 3 (A8 of reservoir) into wells A8-H8 of second plate (third replicate)
    #transfer of 22, 20, 18, 14, 11, 9, 6 and 4 uL of drug 2 stock 3 (A8 of reservoir) into wells A8-H8 of second plate
    P20single.transfer(
        volume = [11, 11, 20, 18, 14, 11, 9, 6, 4],
        source = reservoir["A8"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A8", "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"]],
        new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True) #no need to mix
    
    #3.b transfer of DMSO diluent into first and second columns of each plate for all replicates

    #first columns for first plate all replicates (columns 1, 4 and 7 for first, second and third replicate, respectively)

    #transfer of DMSO (A4 of reservoir) into wells B1-H1 of first plate (first replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A4 of reservoir) into wells B1-H1 of first plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["B1", "C1", "D1", "E1", "F1", "G1", "H1"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A4 of reservoir) into wells B4-H4 of first plate (second replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A4 of reservoir) into wells B4-H4 of first plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["B4", "C4", "D4", "E4", "F4", "G4", "H4"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A4 of reservoir) into wells B7-H7 of first plate (third replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A4 of reservoir) into wells B7-H7 of first plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["B7", "C7", "D7", "E7", "F7", "G7", "H7"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #first columns for second plate all replicates (columns 1, 4 and 7 for first, second and third replicate, respectively)

    #transfer of DMSO (A5 of reservoir) into wells B1-H1 of second plate (first replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A5 of reservoir) into wells B1-H1 of second plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["B1", "C1", "D1", "E1", "F1", "G1", "H1"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A5 of reservoir) into wells B4-H4 of second plate (second replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A5 of reservoir) into wells B4-H4 of second plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["B4", "C4", "D4", "E4", "F4", "G4", "H4"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A5 of reservoir) into wells B7-H7 of second plate (third replicate)
    #transfer of 20, 40, 70, 50, 120, 140 and 160 uL of DMSO (A5 of reservoir) into wells B7-H7 of second plate
    P300single.transfer(
        volume = [20, 40, 70, 50, 120, 140, 160],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["B7", "C7", "D7", "E7", "F7", "G7", "H7"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes

    #second columns (approx. 10x dilution from first column) for first plate all replicates (columns 2, 5 and 8 for first, second and third replicate, respectively)

    #transfer of DMSO (A4 of reservoir) into wells A2-H2 of first plate (first replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A4 of reservoir) into wells A2-H2 of first plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A4 of reservoir) into wells A5-H5 of first plate (second replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A4 of reservoir) into wells A5-H5 of first plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A4 of reservoir) into wells A8-H8 of first plate (third replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A4 of reservoir) into wells A8-H8 of first plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A4"].bottom(z=6),
        dest = [plate1.wells_by_name()[well].bottom(z=2) for well in ["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #second columns (approx. 10x dilution from first column) for second plate all replicates (columns 2, 5 and 8 for first, second and third replicate, respectively)

    #transfer of DMSO (A5 of reservoir) into wells A2-H2 of second plate (first replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A5 of reservoir) into wells A2-H2 of second plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A5 of reservoir) into wells A5-H5 of second plate (second replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A5 of reservoir) into wells A5-H5 of second plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of DMSO (A5 of reservoir) into wells A8-H8 of second plate (third replicate)
    #transfer of 198, 200, 202, 206, 209, 211, 214 and 216 uL of DMSO (A5 of reservoir) into wells A8-H8 of second plate
    P300single.transfer(
        volume = [198, 200, 202, 206, 209, 211, 214, 216],
        source = reservoir["A5"].bottom(z=6),
        dest = [plate2.wells_by_name()[well].bottom(z=2) for well in ["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"]],
        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #3.c serial dilution from wells A2-H2, A5-H5 and A8-H8 with 220 uL to wells A6-H6, A9-H9 and A12-H12, respectively, for all replicates of first and second plate (columns 6 and 9 for second and third replicate, respectively)

    #transfer of 20 uL of 10x dilutions from wells A2-H2, A5-H5 and A8-H8 into wells A3-H3, A6-H6 and A9-H9, respectively, for all replicates of first and second plate (columns 3, 6 and 9 for first, second and third replicate, respectively)
    #single code for all wells of same volume
    #the code for each plate has to be written separately
    #first plate
    #a list has to be written for the source and destination wells
    dest_wells = ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9"] #list of destination wells for first plate
    source_wells = ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"] #list of source wells for first plate
    P20single.transfer(volume = 20, 
                       source = [plate1[well].bottom(z=2) for well in source_wells],
                       dest = [plate1[well].bottom(z=2) for well in dest_wells], 
                       new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True) #rate not written because it has be preset globally for the pipette - set in the beginning of the code
        
    #second plate
    dest_wells = ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9"] #list of destination wells for second plate
    source_wells = ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8"] #list of source wells for second plate
    P20single.transfer(volume = 20,
                       source = [plate2[well].bottom(z=2) for well in source_wells],
                       dest = [plate2[well].bottom(z=2) for well in dest_wells],
                       new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True)
    
    #transfer of 180 uL of DMSO (A4 of reservoir) into wells A3-H3, A6-H6 and A9-H9 of first plate (first, second and third replicate, respectively)
    #single code for all wells of same volume
    #the code for each plate has to be written separately  
    #first plate
    dest_wells = ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9"] #list of destination wells for first plate
    P300single.transfer(volume = 180,
                        source = reservoir["A4"].bottom(z=6),
                        dest = [plate1[well].bottom(z=2) for well in dest_wells],
                        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
                        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #second plate
    dest_wells = ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9"] #list of destination wells for second plate
    P300single.transfer(volume = 180,
                        source = reservoir["A5"].bottom(z=6),
                        dest = [plate2[well].bottom(z=2) for well in dest_wells],
                        mix_after = (3, 220), new_tip = "always", blow_out = True, blowout_location = "destination well", touch_tip = True,
                        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #transfer of 200 uL of DMSO (A4 of reservoir) into wells A10-H10 of first plate for blanks!
    #single code for all wells of same volume
    #the code for each plate has to be written separately  
    #first plate
    dest_wells = ["A10", "B10", "C10", "D10", "E10", "F10", "G10", "H10"] #list of destination wells for first plate
    P300single.transfer(volume = 200,
                        source = reservoir["A4"].bottom(z=6),
                        dest = [plate1[well].bottom(z=2) for well in dest_wells],
                        mix_after = (3, 220), new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True,
                        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #second plate
    dest_wells = ["A10", "B10", "C10", "D10", "E10", "F10", "G10", "H10"] #list of destination wells for second plate
    P300single.transfer(volume = 200,
                        source = reservoir["A5"].bottom(z=6),
                        dest = [plate2[well].bottom(z=2) for well in dest_wells],
                        mix_after = (3, 220), new_tip = "once", blow_out = True, blowout_location = "destination well", touch_tip = True,
                        aspirate_delay=2, dispense_delay=2) #delay of 2 seconds after aspirating and dispensing to allow the liquid to settle and ensure accurate volumes
    
    #protocol completed
    #print statement to indicate protocol completion
    protocol.comment("☀️🌼🌹")