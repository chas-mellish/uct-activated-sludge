Unit StdyUnit;    { steady state program }

interface

 uses Crt,
      VarsUnit,
      IOUnit,
      RateUnit;

 Procedure ScaleValues(Compounds:integer);
 Procedure Newton(Compounds:integer);
 Procedure SteadySolution;
 Procedure StoreAvgConcs;
 Procedure SteadyState;

implementation

var
  Tracer                  : AllReactors;
  TracerConcIn,MassTracer : real;
  Inversions              : integer;

{$I flowss.pas }         { flow distribution for SS case        }
{$I scale.pas }          { scale values for Newton              }
{$I newton.pas }         { solve nonlinear equations            }
{$I waste.pas }          { wastage rate                         }
{$I stdyrslt.pas }       { steady state results                 }

Procedure SteadySolution;

  Procedure Titles;
   Begin
    ClrScr;
    AssignCrt(OutFile);
    Rewrite(OutFile);
    WRITE(OutFile,Chr(201));
    FOR I:=1 TO 76 DO WRITE(OutFile,Chr(205)); WRITELN(OutFile,Chr(187));
    WRITE(OutFile,Chr(186),'              First Reactor              ');
    if LastReactor > 1
           then writeln('     Last Reactor                  ',Chr(186))
           else writeln('                                   ',Chr(186));
    WRITE(OutFile,Chr(186),'   Xbh     Xba      Xs      Ss     ');
    if LastReactor > 1
           then writeln('Sno     Ss      Snh     Sno       MaxF   ',Chr(186))
           else writeln('Sno     Snh     Snd     Alk       MaxF   ',Chr(186));
    WRITE(OutFile,Chr(200));
    FOR I:=1 TO 76 DO WRITE(OutFile,Chr(205)); WRITELN(OutFile,Chr(188));
    writeln;
  End;  (* titles *)

Procedure SeedValues;

  var
   MassXa,MassXe,MassXi,MassXn,Ns,bh24,bn04,frac : real;

    Begin
     (* seed values from steady state model *)
     (* seed values from steady state model *)
     bh24 :=EXP(LN(0.24)+(Temp-20)*LN(1.029));
     bn04 :=EXP(LN(0.04)+(Temp-20)*LN(1.029));
     MassXa := 0.66 * FlowFeed * Sti*(1-fus-fup) * Rs / (1 + bh24 * Rs);
     MassXe := 0.2 * bh24 * Rs * MassXa;
     MassXi := fup * FlowFeed * Sti * Rs;
     Ns := Ixb * (MassXa + MassXe + MassXi) / (Rs * FlowFeed);
     MassXn := FlowFeed * (Nti -Ns -3.0) * Ya * Rs / (1 + bn04 * Rs);
     for k := 1 to LastReactor do
      begin
       frac := Tracer[k] * Vol[k] / MassTracer;
       { Xbh }
       C[k,1] := frac * MassXa / Vol[k];
       { Xe }
       C[k,2] := frac * MassXe / Vol[k];
       { Xba }
       C[k,3] := frac * MassXn / Vol[k];
       { Xs }
       if ReactorAerated[k]
           then C[k,4] := 0.10 * C[k,1]
           else if FracFeed[k] > 0.0
                 then C[k,4] := 0.50 * C[k,1]
                 else C[k,4] := 0.20 * C[k,1];
       { Sbp }
       C[k,5] := C[k,4];
       { Xi }
       C[k,6] := frac * MassXi / Vol[k];
       { Xnd }
       C[k,7] := 10.0;
       { Ss }
       if (( NOT ReactorAerated[k]) AND (FracFeed[k] > 0.0) )
            then C[k,8] := 0.25*C0[8]
            else C[k,8] := 1.0;
       { Snh }
       if ReactorAerated[k] then
                              begin
                               if FracFeed[k] > 0
                                        then C[k,9] :=
                                                     0.5 * VolumeTotal/Vol[k]
                                        else C[k,9] := 0.5;
                              end
                            else C[k,9] := 0.25*(Nti-Ns);
       { Snd }
       C[k,10] := 3.0;
       { Sno }
       if ReactorAerated[k] then C[k,11] := 1.0 * (Nti-Ns)
                            else C[k,11] := 0.001;
       { Salk }
       C[k,12] := 0.5 * C0[12];
       { Snu }
       C[k,13] := fus * Sti;
      end;

     for i := 1 to NoPart do
        C[LastReactor+1,i] := (FlowFeed+FlowRASrecycle-FlowWaste)/FlowRASrecycle
                                * C[LastReactor,i];
     for i := NoPart+1 to LastCompound do
        C[LastReactor+1,i] := C[LastReactor,i];
    End; (* seed values *)

    Begin  (* steady solution *)
      Titles;
      SeedValues;
      AirSupply;
      ScaleValues(LastCompound);
      Newton(LastCompound);
    End; (* steady solution *)


  Procedure StoreAvgConcs;
    Begin
      For k := 1 to LastReactor do
        For i := 1 to (TotalCompounds - 1) do CSteady[k,i]:=C[k,i];
    End; (* store avg concs *)



Procedure SteadyState;

    Begin   { steady state }

      ClrScr;
      WastageAndFlows;
      SteadySolution;
      Window(1,1,80,25);
      AssignCrt(OutFile);
      Rewrite(OutFile);
      TextColor(ResultsCol.Text);
      TextBackground(ResultsCol.Back);
      OutputSteadyState;     { to screen }
      write('*** Hit any key to continue...');
      repeat until KeyPressed;
      Key := ReadKey;
      Close(OutFile);
      SteadyStateResults;    { to printer - on request }
      StoreAvgConcs;

    End;  { steadystate }

end.  { of unit StdyUnit }

