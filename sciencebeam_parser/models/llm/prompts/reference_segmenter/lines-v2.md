The numbered lines below are the reference list of a scientific paper. Each line
is prefixed with its line number and a tab.

Identify where each individual bibliographic reference begins, and return the
line number of each beginning, in order.

A reference BEGINS at its number or marker when it has one. If a line contains
only "12." or "[12]", that line is where reference 12 begins, not the following
line where the author names start.

Do not merge two references into one entry, and do not split one reference
across two entries. A heading such as "References" is not a reference.

Report only bibliographic references. The lines given to you are a region the
upstream segmentation model labelled as the reference list, and it is sometimes
wrong: it can include table rows, figure captions, running headers or body text.
Those are not references, however many of them there are — skip them, and report
only the lines where an actual bibliographic reference begins. If none of the
lines are references, return an empty list.

A bibliographic reference names authors or a title, and usually a year, journal
or publisher. A row of measurements or parameter settings is not a reference even
if it repeats in a regular pattern.
