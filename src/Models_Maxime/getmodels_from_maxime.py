 with open("testmodel.dat", "w") as fout:
    ...:     for ii in range(16):
    ...:         fout.write("{")
    ...:         for jj in range(16):
    ...:             fout.write("{" + data[ii].split("  ")[jj][0] + "," + data[ii].split("  ")[jj][1] + "}, ")
    ...:         fout.write("}\n")