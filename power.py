def MuHatHetero(MuHatHetero20, Temp):
    '''(number, number) -> number

    Caclulate MuHatHetero at a specific temperature

    >>> MuHatHetero(MuHatHetero20, Temp) = 
    '''
    MuHatHetero20 = 1.024
    TempDiff = Temp - 20.0
    return pow(MuHatHetero20, Temp)




'''def TempAdjustment;

   >>>TA.MuHatHetero( 




function Power(B,X : real) : real;
 begin Power := Exp (X * ln(B)); end;

Begin
    TempDiff:=Temp-20.0;
    MuHatHetero:=MuHatHetero20 * Power(ThetaMuHatH,TempDiff);
    Ks:=Ks20 * Power(ThetaKs,TempDiff);
    Bh:=Bh20 * Power(ThetaBh,TempDiff);
    Kmp:=Kmp20 * Power(ThetaKmp,TempDiff);
    Ksp:=Ksp20 * Power(ThetaKsp,TempDiff);
    Ka:=Ka20 * Power(ThetaKa,TempDiff);
    Kr:=Kr20 * Power(ThetaKr,TempDiff);
    MuHatAuto:=MuHatAuto20 * Power(ThetaMuHatA,TempDiff);
    Knh:=Knh20 * Power(ThetaKnh,TempDiff);
    Ba:=Ba20 * Power(ThetaBa,TempDiff);

End;  {* temp adjustment *}


Procedure Heterotrophs;   { check/change heterotroph constants }
 Begin
  FlashScreen;
  ValueDummy[1] := MuHatHetero20  ;
  ValueDummy[2] := Ks20           ;
  ValueDummy[3] := Koh            ;
  ValueDummy[4] := Bh20           ;
  ValueDummy[5] := NetaGrow       ;
  ValueDummy[6] := Kno            ;
  ValueDummy[7] := Kmp20           ;
  ValueDummy[8] := Ksp20           ;
  ValueDummy[9] := Kr20          ;
  ValueDummy[10] := Kna           ;
  ValueDummy[11] := Ka20           ;

  SelectConstant(HeteroList);

  MuHatHetero20  := ValueDummy[1] ;
  Ks20           := ValueDummy[2] ;
  Koh            := ValueDummy[3] ;
  Bh20           := ValueDummy[4] ;
  NetaGrow       := ValueDummy[5] ;
  Kno            := ValueDummy[6] ;
  Kmp20          := ValueDummy[7] ;
  Ksp20          := ValueDummy[8] ;
  Kr20           := ValueDummy[9] ;
  Kna            := ValueDummy[10] ;
  Ka20           := ValueDummy[11] ;
End;  { heterotrophs }


Procedure Autotrophs;   { check/change autotroph constants }
 Begin
  FlashScreen;
  ValueDummy[1] := MuHatAuto20   ;
  ValueDummy[2] := Knh20         ;
  ValueDummy[3] := Koa           ;
  ValueDummy[4] := Ba20          ;

  SelectConstant(AutoList);

  MuHatAuto20  := ValueDummy[1] ;
  Knh20        := ValueDummy[2] ;
  Koa          := ValueDummy[3] ;
  Ba20         := ValueDummy[4] ;

End;  { autotrophs }


Procedure Arrhenius;   { check/change theta values }
 Begin
  FlashScreen;
  ValueDummy[1] := ThetaMuHatH  ;
  ValueDummy[2] := ThetaKs
  ValueDummy[3] := ThetaBh
  ValueDummy[4] := ThetaKmp
  ValueDummy[5] := ThetaKsp
  ValueDummy[6] := ThetaKr
  ValueDummy[7] := ThetaMuHatA
  ValueDummy[8] := ThetaKnh
  ValueDummy[9] := ThetaBa
  ValueDummy[10] := ThetaKa

  SelectConstant(TempList)
  
  ThetaMuHatH       := ValueDummy[1]
  ThetaKs           := ValueDummy[2]
  ThetaBh           := ValueDummy[3]
  ThetaKmp          := ValueDummy[4]
  ThetaKsp          := ValueDummy[5]
  ThetaKr           := ValueDummy[6]
  ThetaMuHatA       := ValueDummy[7]
  ThetaKnh          := ValueDummy[8]
  ThetaBa           := ValueDummy[9]
  ThetaKa           := ValueDummy[10]

End;  { arrhenius }

Kinetics Equations		
                                            Symbol	Typical	ө
Endogenous Respiration Rate (20°C)	    bh20	0.24	1.029
Specfic Endogenous Mass Loss Rate
for Nitrosomonas (20°C)	                    bn20	0.17	1.029
Maximum Specific Growth Rate (20°C)	    mnm20	0.9	1.072
Nitrification Half Satuation Constant(20°C) Kn20	0.7	1
Kinetic Constant for Degradation of
Organic Nitrogen	                    Kr	        0.015	1.029
Denitrification Rate Constant 1	            K1T	        0.720	1.20
Denitrification Rate Constant 2	            K2T	        0.1008	1.08
Denitrification Rate Constant 3	            K3T	        0.072	1.03



Procedure Temperature;
    Begin
      InputBox(MainMenu.CurrentOption);
      Msg(' Operating temperature of plant (degC) = ',2,2);
      Temp := GetReal(8,35);
      MainWindow(MainMenu.CurrentOption);
      WriteIntgrFldRev(2,2,4,1);
      Msg('Process Temperature    degC       =',6,4);
      WriteRealFldDec(Temp,43,4,8,1);
    End;


Parameter	                            Manual	BioWin - Raw	BioWin - Settled
Readily biodegradable (including Acetate)   0.27	0.16	0.27
Acetate	                                    0.15	0.15	0.15
Non-colloidal slowly biodegradable	    0.75	0.75	0.5
Unbiodegradable soluble	                    0.05	0.05	0.08
Unbiodegradable particulate	            0.2	        0.13	0.08
Ammonia	                                    0.66	0.66	0.75
Particulate organic nitrogen	            0.5	        0.5	0.25
Soluble unbiodegradable TKN	            0.02	0.02	0.02
N:COD ratio for unbiodegradable part. COD   0.035	0.035	0.035
Phosphate	                            0.5	        0.5	0.75
P:COD ratio for unbiodegradable part. COD   0.011	0.011	0.011
Non-poly-P heterotrophs	                    0.0001	0.0001	0.0001
Anoxic methanol utilizers	            0.0001	0.0001	0.0001
Ammonia oxidizers	                    0.0001	0.0001	0.0001
Nitrite oxidizers	                    0.0001	0.0001	0.0001
Anaerobic ammonia oxidizers	            0.0001	0.0001	0.0001
PAOs	                                    0.0001	0.0001	0.0001
Propionic acetogens	                    0.0001	0.0001	0.0001
Acetoclastic methanogens	            0.0001	0.0001	0.0001
H2-utilizing methanogens	            0.0001	0.0001	0.0001




'''
