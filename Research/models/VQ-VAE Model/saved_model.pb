¤
ŐŤ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMin

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Ŕ
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68Ó
}
embeddings_vqvaeVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameembeddings_vqvae
v
$embeddings_vqvae/Read/ReadVariableOpReadVariableOpembeddings_vqvae*
_output_shapes
:	*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: *
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
: *
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0

conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameconv2d_transpose_3/kernel

-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:@*
dtype0

conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:@*
dtype0

conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameconv2d_transpose_4/kernel

-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
: @*
dtype0

conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
: *
dtype0

conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_5/kernel

-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
: *
dtype0

conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
 8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ű7
valueŃ7BÎ7 BÇ7
ż
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
 

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

layer-0
layer_with_weights-0
layer-1
 layer_with_weights-1
 layer-2
!layer_with_weights-2
!layer-3
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
b
(0
)1
*2
+3
,4
-5
6
.7
/8
09
110
211
312*
b
(0
)1
*2
+3
,4
-5
6
.7
/8
09
110
211
312*
* 
°
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

9serving_default* 
* 
Ś

(kernel
)bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
Ś

*kernel
+bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*
Ś

,kernel
-bias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses*
.
(0
)1
*2
+3
,4
-5*
.
(0
)1
*2
+3
,4
-5*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
d^
VARIABLE_VALUEembeddings_vqvae:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
Ś

.kernel
/bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses*
Ś

0kernel
1bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*
Ś

2kernel
3bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
.
.0
/1
02
13
24
35*
.
.0
/1
02
13
24
35*
* 

hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUEconv2d_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_8/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_8/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_3/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEconv2d_transpose_3/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d_transpose_4/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*
* 
* 
* 
* 

(0
)1*

(0
)1*
* 

mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
* 
* 

*0
+1*

*0
+1*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 

,0
-1*

,0
-1*
* 

wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 
* 
 
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1*

.0
/1*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 

00
11*

00
11*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 

20
31*

20
31*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
 
0
1
 2
!3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_input_8Placeholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙
á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasembeddings_vqvaeconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_74003
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
š
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$embeddings_vqvae/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_74553
°
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembeddings_vqvaeconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_74602ĺî
Ŕ
ď
#__inference_signature_wrapper_74003
input_8!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
	unknown_5:	#
	unknown_6:@
	unknown_7:@#
	unknown_8: @
	unknown_9: $

unknown_10: 

unknown_11:
identity˘StatefulPartitionedCallĎ
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_72824w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
Ő 

M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_74491

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs

ü
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
˝

0__inference_vector_quantizer_layer_call_fn_74093
x
unknown:	
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallxunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
ˇ

B__inference_encoder_layer_call_and_return_conditional_losses_74061

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource:@6
(conv2d_8_biasadd_readvariableop_resource:
identity˘conv2d_6/BiasAdd/ReadVariableOp˘conv2d_6/Conv2D/ReadVariableOp˘conv2d_7/BiasAdd/ReadVariableOp˘conv2d_7/Conv2D/ReadVariableOp˘conv2d_8/BiasAdd/ReadVariableOp˘conv2d_8/Conv2D/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ť
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ŕ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ŕ
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙p
IdentityIdentityconv2d_8/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
žŻ

A__inference_vq_vae_layer_call_and_return_conditional_losses_73970

inputsI
/encoder_conv2d_6_conv2d_readvariableop_resource: >
0encoder_conv2d_6_biasadd_readvariableop_resource: I
/encoder_conv2d_7_conv2d_readvariableop_resource: @>
0encoder_conv2d_7_biasadd_readvariableop_resource:@I
/encoder_conv2d_8_conv2d_readvariableop_resource:@>
0encoder_conv2d_8_biasadd_readvariableop_resource:B
/vector_quantizer_matmul_readvariableop_resource:	]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity

identity_1˘1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp˘1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp˘1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp˘'encoder/conv2d_6/BiasAdd/ReadVariableOp˘&encoder/conv2d_6/Conv2D/ReadVariableOp˘'encoder/conv2d_7/BiasAdd/ReadVariableOp˘&encoder/conv2d_7/Conv2D/ReadVariableOp˘'encoder/conv2d_8/BiasAdd/ReadVariableOp˘&encoder/conv2d_8/Conv2D/ReadVariableOp˘&vector_quantizer/MatMul/ReadVariableOp˘(vector_quantizer/MatMul_1/ReadVariableOp˘vector_quantizer/ReadVariableOp
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ť
encoder/conv2d_6/Conv2DConv2Dinputs.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ z
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ř
encoder/conv2d_7/Conv2DConv2D#encoder/conv2d_6/Relu:activations:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@z
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
&encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ř
encoder/conv2d_8/Conv2DConv2D#encoder/conv2d_7/Relu:activations:0.encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

'encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
encoder/conv2d_8/BiasAddBiasAdd encoder/conv2d_8/Conv2D:output:0/encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙g
vector_quantizer/ShapeShape!encoder/conv2d_8/BiasAdd:output:0*
T0*
_output_shapes
:o
vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ą
vector_quantizer/ReshapeReshape!encoder/conv2d_8/BiasAdd:output:0'vector_quantizer/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&vector_quantizer/MatMul/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0§
vector_quantizer/MatMulMatMul!vector_quantizer/Reshape:output:0.vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/powPow!vector_quantizer/Reshape:output:0vector_quantizer/pow/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
&vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Š
vector_quantizer/SumSumvector_quantizer/pow:z:0/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
vector_quantizer/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0]
vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_1Pow'vector_quantizer/ReadVariableOp:value:0!vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	j
(vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
vector_quantizer/Sum_1Sumvector_quantizer/pow_1:z:01vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:
vector_quantizer/addAddV2vector_quantizer/Sum:output:0vector_quantizer/Sum_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/mulMulvector_quantizer/mul/x:output:0!vector_quantizer/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/subSubvector_quantizer/add:z:0vector_quantizer/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
vector_quantizer/ArgMinArgMinvector_quantizer/sub:z:0*vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
"vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    a
vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :ů
vector_quantizer/one_hotOneHot vector_quantizer/ArgMin:output:0'vector_quantizer/one_hot/depth:output:0*vector_quantizer/one_hot/on_value:output:0+vector_quantizer/one_hot/off_value:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0˝
vector_quantizer/MatMul_1MatMul!vector_quantizer/one_hot:output:00vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(Ľ
vector_quantizer/Reshape_1Reshape#vector_quantizer/MatMul_1:product:0vector_quantizer/Shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/StopGradientStopGradient#vector_quantizer/Reshape_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
vector_quantizer/sub_1Sub&vector_quantizer/StopGradient:output:0!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙]
vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_2Powvector_quantizer/sub_1:z:0!vector_quantizer/pow_2/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙o
vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             {
vector_quantizer/MeanMeanvector_quantizer/pow_2:z:0vector_quantizer/Const:output:0*
T0*
_output_shapes
: ]
vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  >
vector_quantizer/mul_1Mul!vector_quantizer/mul_1/x:output:0vector_quantizer/Mean:output:0*
T0*
_output_shapes
: 
vector_quantizer/StopGradient_1StopGradient!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
vector_quantizer/sub_2Sub#vector_quantizer/Reshape_1:output:0(vector_quantizer/StopGradient_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙]
vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_3Powvector_quantizer/sub_2:z:0!vector_quantizer/pow_3/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q
vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
vector_quantizer/Mean_1Meanvector_quantizer/pow_3:z:0!vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: ~
vector_quantizer/add_1AddV2vector_quantizer/mul_1:z:0 vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: 
vector_quantizer/sub_3Sub#vector_quantizer/Reshape_1:output:0!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/StopGradient_2StopGradientvector_quantizer/sub_3:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
vector_quantizer/add_2AddV2!encoder/conv2d_8/BiasAdd:output:0(vector_quantizer/StopGradient_2:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
 decoder/conv2d_transpose_3/ShapeShapevector_quantizer/add_2:z:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ż
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
¨
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ř
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@}
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Â
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
¨
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ř
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Â
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
¨
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ř
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Z

Identity_1Identityvector_quantizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp(^encoder/conv2d_8/BiasAdd/ReadVariableOp'^encoder/conv2d_8/Conv2D/ReadVariableOp'^vector_quantizer/MatMul/ReadVariableOp)^vector_quantizer/MatMul_1/ReadVariableOp ^vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp2R
'encoder/conv2d_8/BiasAdd/ReadVariableOp'encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&encoder/conv2d_8/Conv2D/ReadVariableOp&encoder/conv2d_8/Conv2D/ReadVariableOp2P
&vector_quantizer/MatMul/ReadVariableOp&vector_quantizer/MatMul/ReadVariableOp2T
(vector_quantizer/MatMul_1/ReadVariableOp(vector_quantizer/MatMul_1/ReadVariableOp2B
vector_quantizer/ReadVariableOpvector_quantizer/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
Ł
B__inference_encoder_layer_call_and_return_conditional_losses_73016
input_5(
conv2d_6_73000: 
conv2d_6_73002: (
conv2d_7_73005: @
conv2d_7_73007:@(
conv2d_8_73010:@
conv2d_8_73012:
identity˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_8/StatefulPartitionedCallö
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_6_73000conv2d_6_73002*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_73005conv2d_7_73007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_73010conv2d_8_73012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
ˇ

B__inference_encoder_layer_call_and_return_conditional_losses_74085

inputsA
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource: A
'conv2d_7_conv2d_readvariableop_resource: @6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource:@6
(conv2d_8_biasadd_readvariableop_resource:
identity˘conv2d_6/BiasAdd/ReadVariableOp˘conv2d_6/Conv2D/ReadVariableOp˘conv2d_7/BiasAdd/ReadVariableOp˘conv2d_7/Conv2D/ReadVariableOp˘conv2d_8/BiasAdd/ReadVariableOp˘conv2d_8/Conv2D/ReadVariableOp
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ť
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ j
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ŕ
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@j
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ŕ
conv2d_8/Conv2DConv2Dconv2d_7/Relu:activations:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙p
IdentityIdentityconv2d_8/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
˛6
ă
!__inference__traced_restore_74602
file_prefix4
!assignvariableop_embeddings_vqvae:	<
"assignvariableop_1_conv2d_6_kernel: .
 assignvariableop_2_conv2d_6_bias: <
"assignvariableop_3_conv2d_7_kernel: @.
 assignvariableop_4_conv2d_7_bias:@<
"assignvariableop_5_conv2d_8_kernel:@.
 assignvariableop_6_conv2d_8_bias:F
,assignvariableop_7_conv2d_transpose_3_kernel:@8
*assignvariableop_8_conv2d_transpose_3_bias:@F
,assignvariableop_9_conv2d_transpose_4_kernel: @9
+assignvariableop_10_conv2d_transpose_4_bias: G
-assignvariableop_11_conv2d_transpose_5_kernel: 9
+assignvariableop_12_conv2d_transpose_5_bias:
identity_14˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9Ź
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ň
valueČBĹB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_embeddings_vqvaeIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_6_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_6_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_7_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_7_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_8_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_8_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp,assignvariableop_7_conv2d_transpose_3_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp*assignvariableop_8_conv2d_transpose_3_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp,assignvariableop_9_conv2d_transpose_4_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp+assignvariableop_10_conv2d_transpose_4_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp-assignvariableop_11_conv2d_transpose_5_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_5_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¸
˘
B__inference_encoder_layer_call_and_return_conditional_losses_72965

inputs(
conv2d_6_72949: 
conv2d_6_72951: (
conv2d_7_72954: @
conv2d_7_72956:@(
conv2d_8_72959:@
conv2d_8_72961:
identity˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_8/StatefulPartitionedCallő
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_72949conv2d_6_72951*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_72954conv2d_7_72956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_72959conv2d_8_72961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç
ň
&__inference_vq_vae_layer_call_fn_73576
input_8!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
	unknown_5:	#
	unknown_6:@
	unknown_7:@#
	unknown_8: @
	unknown_9: $

unknown_10: 

unknown_11:
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: */
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_vq_vae_layer_call_and_return_conditional_losses_73514w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
Ą	

'__inference_encoder_layer_call_fn_72897
input_5!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72882w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
Ĺ
§
2__inference_conv2d_transpose_5_layer_call_fn_74458

inputs!
unknown: 
	unknown_0:
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
Ő 

M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
¨S
Ď
B__inference_decoder_layer_call_and_return_conditional_losses_74241

inputsU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_3_biasadd_readvariableop_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity˘)conv2d_transpose_3/BiasAdd/ReadVariableOp˘2conv2d_transpose_3/conv2d_transpose/ReadVariableOp˘)conv2d_transpose_4/BiasAdd/ReadVariableOp˘2conv2d_transpose_4/conv2d_transpose/ReadVariableOp˘)conv2d_transpose_5/BiasAdd/ReadVariableOp˘2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@č
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ŕ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : č
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0˘
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ŕ
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :č
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0˘
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ŕ
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙é
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

'__inference_decoder_layer_call_fn_74178

inputs!
unknown:@
	unknown_0:@#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73245w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ż
A__inference_vq_vae_layer_call_and_return_conditional_losses_73611
input_8'
encoder_73579: 
encoder_73581: '
encoder_73583: @
encoder_73585:@'
encoder_73587:@
encoder_73589:)
vector_quantizer_73592:	'
decoder_73596:@
decoder_73598:@'
decoder_73600: @
decoder_73602: '
decoder_73604: 
decoder_73606:
identity

identity_1˘decoder/StatefulPartitionedCall˘encoder/StatefulPartitionedCall˘(vector_quantizer/StatefulPartitionedCallś
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_8encoder_73579encoder_73581encoder_73583encoder_73585encoder_73587encoder_73589*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72882 
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_73592*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386ŕ
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_73596decoder_73598decoder_73600decoder_73602decoder_73604decoder_73606*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73192
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ľ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
č

(__inference_conv2d_7_layer_call_fn_74333

inputs!
unknown: @
	unknown_0:@
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs

Ż
A__inference_vq_vae_layer_call_and_return_conditional_losses_73646
input_8'
encoder_73614: 
encoder_73616: '
encoder_73618: @
encoder_73620:@'
encoder_73622:@
encoder_73624:)
vector_quantizer_73627:	'
decoder_73631:@
decoder_73633:@'
decoder_73635: @
decoder_73637: '
decoder_73639: 
decoder_73641:
identity

identity_1˘decoder/StatefulPartitionedCall˘encoder/StatefulPartitionedCall˘(vector_quantizer/StatefulPartitionedCallś
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_8encoder_73614encoder_73616encoder_73618encoder_73620encoder_73622encoder_73624*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72965 
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_73627*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386ŕ
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_73631decoder_73633decoder_73635decoder_73637decoder_73639decoder_73641*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73245
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ľ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
Ă!

M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
	

'__inference_encoder_layer_call_fn_74037

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72965w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś

ü
C__inference_conv2d_8_layer_call_and_return_conditional_losses_74363

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ĺ
§
2__inference_conv2d_transpose_4_layer_call_fn_74415

inputs!
unknown: @
	unknown_0: 
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ĺ
§
2__inference_conv2d_transpose_3_layer_call_fn_74372

inputs!
unknown:@
	unknown_0:@
identity˘StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
řż
Ś
 __inference__wrapped_model_72824
input_8P
6vq_vae_encoder_conv2d_6_conv2d_readvariableop_resource: E
7vq_vae_encoder_conv2d_6_biasadd_readvariableop_resource: P
6vq_vae_encoder_conv2d_7_conv2d_readvariableop_resource: @E
7vq_vae_encoder_conv2d_7_biasadd_readvariableop_resource:@P
6vq_vae_encoder_conv2d_8_conv2d_readvariableop_resource:@E
7vq_vae_encoder_conv2d_8_biasadd_readvariableop_resource:I
6vq_vae_vector_quantizer_matmul_readvariableop_resource:	d
Jvq_vae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@O
Avq_vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@d
Jvq_vae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @O
Avq_vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource: d
Jvq_vae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: O
Avq_vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity˘8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp˘Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp˘8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp˘Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp˘8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp˘Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp˘.vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOp˘-vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOp˘.vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOp˘-vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOp˘.vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOp˘-vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOp˘-vq_vae/vector_quantizer/MatMul/ReadVariableOp˘/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp˘&vq_vae/vector_quantizer/ReadVariableOpŹ
-vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ę
vq_vae/encoder/conv2d_6/Conv2DConv2Dinput_85vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
˘
.vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ĺ
vq_vae/encoder/conv2d_6/BiasAddBiasAdd'vq_vae/encoder/conv2d_6/Conv2D:output:06vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
vq_vae/encoder/conv2d_6/ReluRelu(vq_vae/encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ Ź
-vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0í
vq_vae/encoder/conv2d_7/Conv2DConv2D*vq_vae/encoder/conv2d_6/Relu:activations:05vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
˘
.vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ĺ
vq_vae/encoder/conv2d_7/BiasAddBiasAdd'vq_vae/encoder/conv2d_7/Conv2D:output:06vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
vq_vae/encoder/conv2d_7/ReluRelu(vq_vae/encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ź
-vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp6vq_vae_encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0í
vq_vae/encoder/conv2d_8/Conv2DConv2D*vq_vae/encoder/conv2d_7/Relu:activations:05vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
˘
.vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp7vq_vae_encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ĺ
vq_vae/encoder/conv2d_8/BiasAddBiasAdd'vq_vae/encoder/conv2d_8/Conv2D:output:06vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙u
vq_vae/vector_quantizer/ShapeShape(vq_vae/encoder/conv2d_8/BiasAdd:output:0*
T0*
_output_shapes
:v
%vq_vae/vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ś
vq_vae/vector_quantizer/ReshapeReshape(vq_vae/encoder/conv2d_8/BiasAdd:output:0.vq_vae/vector_quantizer/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ľ
-vq_vae/vector_quantizer/MatMul/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0ź
vq_vae/vector_quantizer/MatMulMatMul(vq_vae/vector_quantizer/Reshape:output:05vq_vae/vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
vq_vae/vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ś
vq_vae/vector_quantizer/powPow(vq_vae/vector_quantizer/Reshape:output:0&vq_vae/vector_quantizer/pow/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
-vq_vae/vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :ž
vq_vae/vector_quantizer/SumSumvq_vae/vector_quantizer/pow:z:06vq_vae/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
&vq_vae/vector_quantizer/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0d
vq_vae/vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¨
vq_vae/vector_quantizer/pow_1Pow.vq_vae/vector_quantizer/ReadVariableOp:value:0(vq_vae/vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	q
/vq_vae/vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : §
vq_vae/vector_quantizer/Sum_1Sum!vq_vae/vector_quantizer/pow_1:z:08vq_vae/vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:Ľ
vq_vae/vector_quantizer/addAddV2$vq_vae/vector_quantizer/Sum:output:0&vq_vae/vector_quantizer/Sum_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
vq_vae/vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
vq_vae/vector_quantizer/mulMul&vq_vae/vector_quantizer/mul/x:output:0(vq_vae/vector_quantizer/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
vq_vae/vector_quantizer/subSubvq_vae/vector_quantizer/add:z:0vq_vae/vector_quantizer/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(vq_vae/vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :Ş
vq_vae/vector_quantizer/ArgMinArgMinvq_vae/vector_quantizer/sub:z:01vq_vae/vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙m
(vq_vae/vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
)vq_vae/vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    h
%vq_vae/vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :
vq_vae/vector_quantizer/one_hotOneHot'vq_vae/vector_quantizer/ArgMin:output:0.vq_vae/vector_quantizer/one_hot/depth:output:01vq_vae/vector_quantizer/one_hot/on_value:output:02vq_vae/vector_quantizer/one_hot/off_value:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙§
/vq_vae/vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp6vq_vae_vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ň
 vq_vae/vector_quantizer/MatMul_1MatMul(vq_vae/vector_quantizer/one_hot:output:07vq_vae/vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(ş
!vq_vae/vector_quantizer/Reshape_1Reshape*vq_vae/vector_quantizer/MatMul_1:product:0&vq_vae/vector_quantizer/Shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
$vq_vae/vector_quantizer/StopGradientStopGradient*vq_vae/vector_quantizer/Reshape_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ˇ
vq_vae/vector_quantizer/sub_1Sub-vq_vae/vector_quantizer/StopGradient:output:0(vq_vae/encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d
vq_vae/vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ť
vq_vae/vector_quantizer/pow_2Pow!vq_vae/vector_quantizer/sub_1:z:0(vq_vae/vector_quantizer/pow_2/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙v
vq_vae/vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             
vq_vae/vector_quantizer/MeanMean!vq_vae/vector_quantizer/pow_2:z:0&vq_vae/vector_quantizer/Const:output:0*
T0*
_output_shapes
: d
vq_vae/vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  >
vq_vae/vector_quantizer/mul_1Mul(vq_vae/vector_quantizer/mul_1/x:output:0%vq_vae/vector_quantizer/Mean:output:0*
T0*
_output_shapes
: 
&vq_vae/vector_quantizer/StopGradient_1StopGradient(vq_vae/encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
vq_vae/vector_quantizer/sub_2Sub*vq_vae/vector_quantizer/Reshape_1:output:0/vq_vae/vector_quantizer/StopGradient_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d
vq_vae/vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ť
vq_vae/vector_quantizer/pow_3Pow!vq_vae/vector_quantizer/sub_2:z:0(vq_vae/vector_quantizer/pow_3/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x
vq_vae/vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
vq_vae/vector_quantizer/Mean_1Mean!vq_vae/vector_quantizer/pow_3:z:0(vq_vae/vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: 
vq_vae/vector_quantizer/add_1AddV2!vq_vae/vector_quantizer/mul_1:z:0'vq_vae/vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: ´
vq_vae/vector_quantizer/sub_3Sub*vq_vae/vector_quantizer/Reshape_1:output:0(vq_vae/encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
&vq_vae/vector_quantizer/StopGradient_2StopGradient!vq_vae/vector_quantizer/sub_3:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ť
vq_vae/vector_quantizer/add_2AddV2(vq_vae/encoder/conv2d_8/BiasAdd:output:0/vq_vae/vector_quantizer/StopGradient_2:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙x
'vq_vae/decoder/conv2d_transpose_3/ShapeShape!vq_vae/vector_quantizer/add_2:z:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/vq_vae/decoder/conv2d_transpose_3/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_3/Shape:output:0>vq_vae/decoder/conv2d_transpose_3/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@ł
'vq_vae/decoder/conv2d_transpose_3/stackPack8vq_vae/decoder/conv2d_transpose_3/strided_slice:output:02vq_vae/decoder/conv2d_transpose_3/stack/1:output:02vq_vae/decoder/conv2d_transpose_3/stack/2:output:02vq_vae/decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:
7vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1vq_vae/decoder/conv2d_transpose_3/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_3/stack:output:0@vq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÔ
Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ë
2vq_vae/decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_3/stack:output:0Ivq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0!vq_vae/vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
ś
8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0í
)vq_vae/decoder/conv2d_transpose_3/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_3/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
&vq_vae/decoder/conv2d_transpose_3/ReluRelu2vq_vae/decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
'vq_vae/decoder/conv2d_transpose_4/ShapeShape4vq_vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/vq_vae/decoder/conv2d_transpose_4/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_4/Shape:output:0>vq_vae/decoder/conv2d_transpose_4/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ł
'vq_vae/decoder/conv2d_transpose_4/stackPack8vq_vae/decoder/conv2d_transpose_4/strided_slice:output:02vq_vae/decoder/conv2d_transpose_4/stack/1:output:02vq_vae/decoder/conv2d_transpose_4/stack/2:output:02vq_vae/decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:
7vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1vq_vae/decoder/conv2d_transpose_4/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_4/stack:output:0@vq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÔ
Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ţ
2vq_vae/decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_4/stack:output:0Ivq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
ś
8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0í
)vq_vae/decoder/conv2d_transpose_4/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_4/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
&vq_vae/decoder/conv2d_transpose_4/ReluRelu2vq_vae/decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
'vq_vae/decoder/conv2d_transpose_5/ShapeShape4vq_vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:
5vq_vae/decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/vq_vae/decoder/conv2d_transpose_5/strided_sliceStridedSlice0vq_vae/decoder/conv2d_transpose_5/Shape:output:0>vq_vae/decoder/conv2d_transpose_5/strided_slice/stack:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_1:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)vq_vae/decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :k
)vq_vae/decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ł
'vq_vae/decoder/conv2d_transpose_5/stackPack8vq_vae/decoder/conv2d_transpose_5/strided_slice:output:02vq_vae/decoder/conv2d_transpose_5/stack/1:output:02vq_vae/decoder/conv2d_transpose_5/stack/2:output:02vq_vae/decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:
7vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1vq_vae/decoder/conv2d_transpose_5/strided_slice_1StridedSlice0vq_vae/decoder/conv2d_transpose_5/stack:output:0@vq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack:output:0Bvq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0Bvq_vae/decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÔ
Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpJvq_vae_decoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Ţ
2vq_vae/decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput0vq_vae/decoder/conv2d_transpose_5/stack:output:0Ivq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:04vq_vae/decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
ś
8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpAvq_vae_decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0í
)vq_vae/decoder/conv2d_transpose_5/BiasAddBiasAdd;vq_vae/decoder/conv2d_transpose_5/conv2d_transpose:output:0@vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity2vq_vae/decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ń
NoOpNoOp9^vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpB^vq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp/^vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOp/^vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOp.^vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOp.^vq_vae/vector_quantizer/MatMul/ReadVariableOp0^vq_vae/vector_quantizer/MatMul_1/ReadVariableOp'^vq_vae/vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2t
8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2
Avq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2
Avq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp8vq_vae/decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2
Avq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpAvq_vae/decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2`
.vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_6/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_6/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_7/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_7/Conv2D/ReadVariableOp2`
.vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOp.vq_vae/encoder/conv2d_8/BiasAdd/ReadVariableOp2^
-vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOp-vq_vae/encoder/conv2d_8/Conv2D/ReadVariableOp2^
-vq_vae/vector_quantizer/MatMul/ReadVariableOp-vq_vae/vector_quantizer/MatMul/ReadVariableOp2b
/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp/vq_vae/vector_quantizer/MatMul_1/ReadVariableOp2P
&vq_vae/vector_quantizer/ReadVariableOp&vq_vae/vector_quantizer/ReadVariableOp:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
	

'__inference_decoder_layer_call_fn_74161

inputs!
unknown:@
	unknown_0:@#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73192w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
C__inference_conv2d_6_layer_call_and_return_conditional_losses_74324

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ä
ń
&__inference_vq_vae_layer_call_fn_73678

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
	unknown_5:	#
	unknown_6:@
	unknown_7:@#
	unknown_8: @
	unknown_9: $

unknown_10: 

unknown_11:
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: */
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_vq_vae_layer_call_and_return_conditional_losses_73406w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
č

(__inference_conv2d_6_layer_call_fn_74313

inputs!
unknown: 
	unknown_0: 
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ą	

'__inference_decoder_layer_call_fn_73207
input_7!
unknown:@
	unknown_0:@#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73192w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7

ý
B__inference_decoder_layer_call_and_return_conditional_losses_73296
input_72
conv2d_transpose_3_73280:@&
conv2d_transpose_3_73282:@2
conv2d_transpose_4_73285: @&
conv2d_transpose_4_73287: 2
conv2d_transpose_5_73290: &
conv2d_transpose_5_73292:
identity˘*conv2d_transpose_3/StatefulPartitionedCall˘*conv2d_transpose_4/StatefulPartitionedCall˘*conv2d_transpose_5/StatefulPartitionedCall
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_transpose_3_73280conv2d_transpose_3_73282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073Ę
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_73285conv2d_transpose_4_73287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118Ę
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_73290conv2d_transpose_5_73292*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7
Ă!

M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ý
B__inference_decoder_layer_call_and_return_conditional_losses_73315
input_72
conv2d_transpose_3_73299:@&
conv2d_transpose_3_73301:@2
conv2d_transpose_4_73304: @&
conv2d_transpose_4_73306: 2
conv2d_transpose_5_73309: &
conv2d_transpose_5_73311:
identity˘*conv2d_transpose_3/StatefulPartitionedCall˘*conv2d_transpose_4/StatefulPartitionedCall˘*conv2d_transpose_5/StatefulPartitionedCall
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinput_7conv2d_transpose_3_73299conv2d_transpose_3_73301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073Ę
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_73304conv2d_transpose_4_73306*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118Ę
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_73309conv2d_transpose_5_73311*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7
Ą	

'__inference_encoder_layer_call_fn_72997
input_5!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72965w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
ä
ń
&__inference_vq_vae_layer_call_fn_73710

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
	unknown_5:	#
	unknown_6:@
	unknown_7:@#
	unknown_8: @
	unknown_9: $

unknown_10: 

unknown_11:
identity˘StatefulPartitionedCallň
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: */
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_vq_vae_layer_call_and_return_conditional_losses_73514w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

Ž
A__inference_vq_vae_layer_call_and_return_conditional_losses_73406

inputs'
encoder_73322: 
encoder_73324: '
encoder_73326: @
encoder_73328:@'
encoder_73330:@
encoder_73332:)
vector_quantizer_73387:	'
decoder_73391:@
decoder_73393:@'
decoder_73395: @
decoder_73397: '
decoder_73399: 
decoder_73401:
identity

identity_1˘decoder/StatefulPartitionedCall˘encoder/StatefulPartitionedCall˘(vector_quantizer/StatefulPartitionedCallľ
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_73322encoder_73324encoder_73326encoder_73328encoder_73330encoder_73332*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72882 
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_73387*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386ŕ
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_73391decoder_73393decoder_73395decoder_73397decoder_73399decoder_73401*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73192
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ľ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
)
ë
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386
x1
matmul_readvariableop_resource:	
identity

identity_1˘MatMul/ReadVariableOp˘MatMul_1/ReadVariableOp˘ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   _
ReshapeReshapexReshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0t
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
powPowReshape:output:0pow/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(n
ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
pow_1PowReadVariableOp:value:0pow_1/y:output:0*
T0*
_output_shapes
:	Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : _
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:]
addAddV2Sum:output:0Sum_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
mulMulmul/x:output:0MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
subSubadd:z:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :b
ArgMinArgMinsub:z:0ArgMin/dimension:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    P
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :¤
one_hotOneHotArgMin:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
MatMul_1MatMulone_hot:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(r
	Reshape_1ReshapeMatMul_1:product:0Shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
StopGradientStopGradientReshape_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
sub_1SubStopGradient:output:0x*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙L
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
pow_2Pow	sub_1:z:0pow_2/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
MeanMean	pow_2:z:0Const:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  >N
mul_1Mulmul_1/x:output:0Mean:output:0*
T0*
_output_shapes
: [
StopGradient_1StopGradientx*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙s
sub_2SubReshape_1:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
pow_3Pow	sub_2:z:0pow_3/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             L
Mean_1Mean	pow_3:z:0Const_1:output:0*
T0*
_output_shapes
: K
add_1AddV2	mul_1:z:0Mean_1:output:0*
T0*
_output_shapes
: ]
sub_3SubReshape_1:output:0x*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙c
StopGradient_2StopGradient	sub_3:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d
add_2AddV2xStopGradient_2:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentity	add_2:z:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙I

Identity_1Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:R N
/
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
Ś

ü
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

ü
B__inference_decoder_layer_call_and_return_conditional_losses_73245

inputs2
conv2d_transpose_3_73229:@&
conv2d_transpose_3_73231:@2
conv2d_transpose_4_73234: @&
conv2d_transpose_4_73236: 2
conv2d_transpose_5_73239: &
conv2d_transpose_5_73241:
identity˘*conv2d_transpose_3/StatefulPartitionedCall˘*conv2d_transpose_4/StatefulPartitionedCall˘*conv2d_transpose_5/StatefulPartitionedCall
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_73229conv2d_transpose_3_73231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073Ę
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_73234conv2d_transpose_4_73236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118Ę
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_73239conv2d_transpose_5_73241*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ü
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
)
ë
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_74144
x1
matmul_readvariableop_resource:	
identity

identity_1˘MatMul/ReadVariableOp˘MatMul_1/ReadVariableOp˘ReadVariableOp6
ShapeShapex*
T0*
_output_shapes
:^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   _
ReshapeReshapexReshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0t
MatMulMatMulReshape:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @^
powPowReshape:output:0pow/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
SumSumpow:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(n
ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
pow_1PowReadVariableOp:value:0pow_1/y:output:0*
T0*
_output_shapes
:	Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : _
Sum_1Sum	pow_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:]
addAddV2Sum:output:0Sum_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
mulMulmul/x:output:0MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙O
subSubadd:z:0mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :b
ArgMinArgMinsub:z:0ArgMin/dimension:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    P
one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :¤
one_hotOneHotArgMin:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
MatMul_1MatMulone_hot:output:0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(r
	Reshape_1ReshapeMatMul_1:product:0Shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
StopGradientStopGradientReshape_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
sub_1SubStopGradient:output:0x*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙L
pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
pow_2Pow	sub_1:z:0pow_2/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙^
ConstConst*
_output_shapes
:*
dtype0*%
valueB"             H
MeanMean	pow_2:z:0Const:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  >N
mul_1Mulmul_1/x:output:0Mean:output:0*
T0*
_output_shapes
: [
StopGradient_1StopGradientx*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙s
sub_2SubReshape_1:output:0StopGradient_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @c
pow_3Pow	sub_2:z:0pow_3/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             L
Mean_1Mean	pow_3:z:0Const_1:output:0*
T0*
_output_shapes
: K
add_1AddV2	mul_1:z:0Mean_1:output:0*
T0*
_output_shapes
: ]
sub_3SubReshape_1:output:0x*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙c
StopGradient_2StopGradient	sub_3:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙d
add_2AddV2xStopGradient_2:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
IdentityIdentity	add_2:z:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙I

Identity_1Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:˙˙˙˙˙˙˙˙˙: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:R N
/
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namex
žŻ

A__inference_vq_vae_layer_call_and_return_conditional_losses_73840

inputsI
/encoder_conv2d_6_conv2d_readvariableop_resource: >
0encoder_conv2d_6_biasadd_readvariableop_resource: I
/encoder_conv2d_7_conv2d_readvariableop_resource: @>
0encoder_conv2d_7_biasadd_readvariableop_resource:@I
/encoder_conv2d_8_conv2d_readvariableop_resource:@>
0encoder_conv2d_8_biasadd_readvariableop_resource:B
/vector_quantizer_matmul_readvariableop_resource:	]
Cdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@H
:decoder_conv2d_transpose_3_biasadd_readvariableop_resource:@]
Cdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @H
:decoder_conv2d_transpose_4_biasadd_readvariableop_resource: ]
Cdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource: H
:decoder_conv2d_transpose_5_biasadd_readvariableop_resource:
identity

identity_1˘1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp˘1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp˘1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp˘:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp˘'encoder/conv2d_6/BiasAdd/ReadVariableOp˘&encoder/conv2d_6/Conv2D/ReadVariableOp˘'encoder/conv2d_7/BiasAdd/ReadVariableOp˘&encoder/conv2d_7/Conv2D/ReadVariableOp˘'encoder/conv2d_8/BiasAdd/ReadVariableOp˘&encoder/conv2d_8/Conv2D/ReadVariableOp˘&vector_quantizer/MatMul/ReadVariableOp˘(vector_quantizer/MatMul_1/ReadVariableOp˘vector_quantizer/ReadVariableOp
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ť
encoder/conv2d_6/Conv2DConv2Dinputs.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ z
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ř
encoder/conv2d_7/Conv2DConv2D#encoder/conv2d_6/Relu:activations:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0°
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@z
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
&encoder/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ř
encoder/conv2d_8/Conv2DConv2D#encoder/conv2d_7/Relu:activations:0.encoder/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

'encoder/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
encoder/conv2d_8/BiasAddBiasAdd encoder/conv2d_8/Conv2D:output:0/encoder/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙g
vector_quantizer/ShapeShape!encoder/conv2d_8/BiasAdd:output:0*
T0*
_output_shapes
:o
vector_quantizer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ą
vector_quantizer/ReshapeReshape!encoder/conv2d_8/BiasAdd:output:0'vector_quantizer/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
&vector_quantizer/MatMul/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0§
vector_quantizer/MatMulMatMul!vector_quantizer/Reshape:output:0.vector_quantizer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
vector_quantizer/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/powPow!vector_quantizer/Reshape:output:0vector_quantizer/pow/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
&vector_quantizer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Š
vector_quantizer/SumSumvector_quantizer/pow:z:0/vector_quantizer/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
vector_quantizer/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0]
vector_quantizer/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_1Pow'vector_quantizer/ReadVariableOp:value:0!vector_quantizer/pow_1/y:output:0*
T0*
_output_shapes
:	j
(vector_quantizer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
vector_quantizer/Sum_1Sumvector_quantizer/pow_1:z:01vector_quantizer/Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:
vector_quantizer/addAddV2vector_quantizer/Sum:output:0vector_quantizer/Sum_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
vector_quantizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/mulMulvector_quantizer/mul/x:output:0!vector_quantizer/MatMul:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/subSubvector_quantizer/add:z:0vector_quantizer/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!vector_quantizer/ArgMin/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
vector_quantizer/ArgMinArgMinvector_quantizer/sub:z:0*vector_quantizer/ArgMin/dimension:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!vector_quantizer/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
"vector_quantizer/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    a
vector_quantizer/one_hot/depthConst*
_output_shapes
: *
dtype0*
value
B :ů
vector_quantizer/one_hotOneHot vector_quantizer/ArgMin:output:0'vector_quantizer/one_hot/depth:output:0*vector_quantizer/one_hot/on_value:output:0+vector_quantizer/one_hot/off_value:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(vector_quantizer/MatMul_1/ReadVariableOpReadVariableOp/vector_quantizer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0˝
vector_quantizer/MatMul_1MatMul!vector_quantizer/one_hot:output:00vector_quantizer/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(Ľ
vector_quantizer/Reshape_1Reshape#vector_quantizer/MatMul_1:product:0vector_quantizer/Shape:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/StopGradientStopGradient#vector_quantizer/Reshape_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
vector_quantizer/sub_1Sub&vector_quantizer/StopGradient:output:0!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙]
vector_quantizer/pow_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_2Powvector_quantizer/sub_1:z:0!vector_quantizer/pow_2/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙o
vector_quantizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             {
vector_quantizer/MeanMeanvector_quantizer/pow_2:z:0vector_quantizer/Const:output:0*
T0*
_output_shapes
: ]
vector_quantizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  >
vector_quantizer/mul_1Mul!vector_quantizer/mul_1/x:output:0vector_quantizer/Mean:output:0*
T0*
_output_shapes
: 
vector_quantizer/StopGradient_1StopGradient!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
vector_quantizer/sub_2Sub#vector_quantizer/Reshape_1:output:0(vector_quantizer/StopGradient_1:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙]
vector_quantizer/pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
vector_quantizer/pow_3Powvector_quantizer/sub_2:z:0!vector_quantizer/pow_3/y:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q
vector_quantizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             
vector_quantizer/Mean_1Meanvector_quantizer/pow_3:z:0!vector_quantizer/Const_1:output:0*
T0*
_output_shapes
: ~
vector_quantizer/add_1AddV2vector_quantizer/mul_1:z:0 vector_quantizer/Mean_1:output:0*
T0*
_output_shapes
: 
vector_quantizer/sub_3Sub#vector_quantizer/Reshape_1:output:0!encoder/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
vector_quantizer/StopGradient_2StopGradientvector_quantizer/sub_3:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
vector_quantizer/add_2AddV2!encoder/conv2d_8/BiasAdd:output:0(vector_quantizer/StopGradient_2:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙j
 decoder/conv2d_transpose_3/ShapeShapevector_quantizer/add_2:z:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_3/strided_sliceStridedSlice)decoder/conv2d_transpose_3/Shape:output:07decoder/conv2d_transpose_3/strided_slice/stack:output:09decoder/conv2d_transpose_3/strided_slice/stack_1:output:09decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@
 decoder/conv2d_transpose_3/stackPack1decoder/conv2d_transpose_3/strided_slice:output:0+decoder/conv2d_transpose_3/stack/1:output:0+decoder/conv2d_transpose_3/stack/2:output:0+decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_3/strided_slice_1StridedSlice)decoder/conv2d_transpose_3/stack:output:09decoder/conv2d_transpose_3/strided_slice_1/stack:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ż
+decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_3/stack:output:0Bdecoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0vector_quantizer/add_2:z:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
¨
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ř
"decoder/conv2d_transpose_3/BiasAddBiasAdd4decoder/conv2d_transpose_3/conv2d_transpose:output:09decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
decoder/conv2d_transpose_3/ReluRelu+decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@}
 decoder/conv2d_transpose_4/ShapeShape-decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_4/strided_sliceStridedSlice)decoder/conv2d_transpose_4/Shape:output:07decoder/conv2d_transpose_4/strided_slice/stack:output:09decoder/conv2d_transpose_4/strided_slice/stack_1:output:09decoder/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 
 decoder/conv2d_transpose_4/stackPack1decoder/conv2d_transpose_4/strided_slice:output:0+decoder/conv2d_transpose_4/stack/1:output:0+decoder/conv2d_transpose_4/stack/2:output:0+decoder/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_4/strided_slice_1StridedSlice)decoder/conv2d_transpose_4/stack:output:09decoder/conv2d_transpose_4/strided_slice_1/stack:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Â
+decoder/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_4/stack:output:0Bdecoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
¨
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ř
"decoder/conv2d_transpose_4/BiasAddBiasAdd4decoder/conv2d_transpose_4/conv2d_transpose:output:09decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
decoder/conv2d_transpose_4/ReluRelu+decoder/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ }
 decoder/conv2d_transpose_5/ShapeShape-decoder/conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:x
.decoder/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(decoder/conv2d_transpose_5/strided_sliceStridedSlice)decoder/conv2d_transpose_5/Shape:output:07decoder/conv2d_transpose_5/strided_slice/stack:output:09decoder/conv2d_transpose_5/strided_slice/stack_1:output:09decoder/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"decoder/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"decoder/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
 decoder/conv2d_transpose_5/stackPack1decoder/conv2d_transpose_5/strided_slice:output:0+decoder/conv2d_transpose_5/stack/1:output:0+decoder/conv2d_transpose_5/stack/2:output:0+decoder/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:z
0decoder/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2decoder/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2decoder/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ŕ
*decoder/conv2d_transpose_5/strided_slice_1StridedSlice)decoder/conv2d_transpose_5/stack:output:09decoder/conv2d_transpose_5/strided_slice_1/stack:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskĆ
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0Â
+decoder/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_5/stack:output:0Bdecoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
¨
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ř
"decoder/conv2d_transpose_5/BiasAddBiasAdd4decoder/conv2d_transpose_5/conv2d_transpose:output:09decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙
IdentityIdentity+decoder/conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Z

Identity_1Identityvector_quantizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp2^decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp(^encoder/conv2d_8/BiasAdd/ReadVariableOp'^encoder/conv2d_8/Conv2D/ReadVariableOp'^vector_quantizer/MatMul/ReadVariableOp)^vector_quantizer/MatMul_1/ReadVariableOp ^vector_quantizer/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2f
1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_4/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp2R
'encoder/conv2d_8/BiasAdd/ReadVariableOp'encoder/conv2d_8/BiasAdd/ReadVariableOp2P
&encoder/conv2d_8/Conv2D/ReadVariableOp&encoder/conv2d_8/Conv2D/ReadVariableOp2P
&vector_quantizer/MatMul/ReadVariableOp&vector_quantizer/MatMul/ReadVariableOp2T
(vector_quantizer/MatMul_1/ReadVariableOp(vector_quantizer/MatMul_1/ReadVariableOp2B
vector_quantizer/ReadVariableOpvector_quantizer/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă!

M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_74406

inputsB
(conv2d_transpose_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@{
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
%

__inference__traced_save_74553
file_prefix/
+savev2_embeddings_vqvae_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Š
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ň
valueČBĹB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B Ş
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_embeddings_vqvae_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*´
_input_shapes˘
: :	: : : @:@:@::@:@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
::,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: 
ç
ň
&__inference_vq_vae_layer_call_fn_73436
input_8!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
	unknown_5:	#
	unknown_6:@
	unknown_7:@#
	unknown_8: @
	unknown_9: $

unknown_10: 

unknown_11:
identity˘StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: */
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_vq_vae_layer_call_and_return_conditional_losses_73406w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
¨S
Ď
B__inference_decoder_layer_call_and_return_conditional_losses_74304

inputsU
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:@@
2conv2d_transpose_3_biasadd_readvariableop_resource:@U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource: @@
2conv2d_transpose_4_biasadd_readvariableop_resource: U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity˘)conv2d_transpose_3/BiasAdd/ReadVariableOp˘2conv2d_transpose_3/conv2d_transpose/ReadVariableOp˘)conv2d_transpose_4/BiasAdd/ReadVariableOp˘2conv2d_transpose_4/conv2d_transpose/ReadVariableOp˘)conv2d_transpose_5/BiasAdd/ReadVariableOp˘2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_3/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@č
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype0
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ŕ
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@~
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@m
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B : č
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0˘
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ŕ
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ ~
conv2d_transpose_4/ReluRelu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ m
conv2d_transpose_5/ShapeShape%conv2d_transpose_4/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :č
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskś
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0˘
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_4/Relu:activations:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides

)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ŕ
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙z
IdentityIdentity#conv2d_transpose_5/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙é
NoOpNoOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ă!

M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_74449

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity˘BiasAdd/ReadVariableOp˘conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ů
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

ü
C__inference_conv2d_7_layer_call_and_return_conditional_losses_74344

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity˘BiasAdd/ReadVariableOp˘Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
 
_user_specified_nameinputs
č

(__inference_conv2d_8_layer_call_fn_74353

inputs!
unknown:@
	unknown_0:
identity˘StatefulPartitionedCallŕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs

ü
B__inference_decoder_layer_call_and_return_conditional_losses_73192

inputs2
conv2d_transpose_3_73176:@&
conv2d_transpose_3_73178:@2
conv2d_transpose_4_73181: @&
conv2d_transpose_4_73183: 2
conv2d_transpose_5_73186: &
conv2d_transpose_5_73188:
identity˘*conv2d_transpose_3/StatefulPartitionedCall˘*conv2d_transpose_4/StatefulPartitionedCall˘*conv2d_transpose_5/StatefulPartitionedCall
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_3_73176conv2d_transpose_3_73178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_73073Ę
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_73181conv2d_transpose_4_73183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_73118Ę
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_73186conv2d_transpose_5_73188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_73162
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
NoOpNoOp+^conv2d_transpose_3/StatefulPartitionedCall+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť
Ł
B__inference_encoder_layer_call_and_return_conditional_losses_73035
input_5(
conv2d_6_73019: 
conv2d_6_73021: (
conv2d_7_73024: @
conv2d_7_73026:@(
conv2d_8_73029:@
conv2d_8_73031:
identity˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_8/StatefulPartitionedCallö
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_6_73019conv2d_6_73021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_73024conv2d_7_73026*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_73029conv2d_8_73031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_5
Ą	

'__inference_decoder_layer_call_fn_73277
input_7!
unknown:@
	unknown_0:@#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73245w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7

Ž
A__inference_vq_vae_layer_call_and_return_conditional_losses_73514

inputs'
encoder_73482: 
encoder_73484: '
encoder_73486: @
encoder_73488:@'
encoder_73490:@
encoder_73492:)
vector_quantizer_73495:	'
decoder_73499:@
decoder_73501:@'
decoder_73503: @
decoder_73505: '
decoder_73507: 
decoder_73509:
identity

identity_1˘decoder/StatefulPartitionedCall˘encoder/StatefulPartitionedCall˘(vector_quantizer/StatefulPartitionedCallľ
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_73482encoder_73484encoder_73486encoder_73488encoder_73490encoder_73492*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72965 
(vector_quantizer/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0vector_quantizer_73495*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:˙˙˙˙˙˙˙˙˙: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_73386ŕ
decoder/StatefulPartitionedCallStatefulPartitionedCall1vector_quantizer/StatefulPartitionedCall:output:0decoder_73499decoder_73501decoder_73503decoder_73505decoder_73507decoder_73509*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_73245
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙q

Identity_1Identity1vector_quantizer/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ľ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall)^vector_quantizer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:˙˙˙˙˙˙˙˙˙: : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall2T
(vector_quantizer/StatefulPartitionedCall(vector_quantizer/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	

'__inference_encoder_layer_call_fn_74020

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_72882w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
¸
˘
B__inference_encoder_layer_call_and_return_conditional_losses_72882

inputs(
conv2d_6_72843: 
conv2d_6_72845: (
conv2d_7_72860: @
conv2d_7_72862:@(
conv2d_8_72876:@
conv2d_8_72878:
identity˘ conv2d_6/StatefulPartitionedCall˘ conv2d_7/StatefulPartitionedCall˘ conv2d_8/StatefulPartitionedCallő
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_72843conv2d_6_72845*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_6_layer_call_and_return_conditional_losses_72842
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_72860conv2d_7_72862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_72859
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0conv2d_8_72876conv2d_8_72878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_72875
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙Ż
NoOpNoOp!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙: : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"ŰL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ş
serving_defaultŚ
C
input_88
serving_default_input_8:0˙˙˙˙˙˙˙˙˙C
decoder8
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:˝ą
Ö
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Š
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
ľ

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Š
layer-0
layer_with_weights-0
layer-1
 layer_with_weights-1
 layer-2
!layer_with_weights-2
!layer-3
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_network
~
(0
)1
*2
+3
,4
-5
6
.7
/8
09
110
211
312"
trackable_list_wrapper
~
(0
)1
*2
+3
,4
-5
6
.7
/8
09
110
211
312"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ć2ă
&__inference_vq_vae_layer_call_fn_73436
&__inference_vq_vae_layer_call_fn_73678
&__inference_vq_vae_layer_call_fn_73710
&__inference_vq_vae_layer_call_fn_73576Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ň2Ď
A__inference_vq_vae_layer_call_and_return_conditional_losses_73840
A__inference_vq_vae_layer_call_and_return_conditional_losses_73970
A__inference_vq_vae_layer_call_and_return_conditional_losses_73611
A__inference_vq_vae_layer_call_and_return_conditional_losses_73646Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ËBČ
 __inference__wrapped_model_72824input_8"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
9serving_default"
signature_map
"
_tf_keras_input_layer
ť

(kernel
)bias
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

*kernel
+bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

,kernel
-bias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses"
_tf_keras_layer
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ę2ç
'__inference_encoder_layer_call_fn_72897
'__inference_encoder_layer_call_fn_74020
'__inference_encoder_layer_call_fn_74037
'__inference_encoder_layer_call_fn_72997Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
B__inference_encoder_layer_call_and_return_conditional_losses_74061
B__inference_encoder_layer_call_and_return_conditional_losses_74085
B__inference_encoder_layer_call_and_return_conditional_losses_73016
B__inference_encoder_layer_call_and_return_conditional_losses_73035Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
#:!	2embeddings_vqvae
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ő2Ň
0__inference_vector_quantizer_layer_call_fn_74093
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_74144
˛
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
"
_tf_keras_input_layer
ť

.kernel
/bias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

0kernel
1bias
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
ť

2kernel
3bias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
J
.0
/1
02
13
24
35"
trackable_list_wrapper
J
.0
/1
02
13
24
35"
trackable_list_wrapper
 "
trackable_list_wrapper
­
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ę2ç
'__inference_decoder_layer_call_fn_73207
'__inference_decoder_layer_call_fn_74161
'__inference_decoder_layer_call_fn_74178
'__inference_decoder_layer_call_fn_73277Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ö2Ó
B__inference_decoder_layer_call_and_return_conditional_losses_74241
B__inference_decoder_layer_call_and_return_conditional_losses_74304
B__inference_decoder_layer_call_and_return_conditional_losses_73296
B__inference_decoder_layer_call_and_return_conditional_losses_73315Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
):' 2conv2d_6/kernel
: 2conv2d_6/bias
):' @2conv2d_7/kernel
:@2conv2d_7/bias
):'@2conv2d_8/kernel
:2conv2d_8/bias
3:1@2conv2d_transpose_3/kernel
%:#@2conv2d_transpose_3/bias
3:1 @2conv2d_transpose_4/kernel
%:# 2conv2d_transpose_4/bias
3:1 2conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ĘBÇ
#__inference_signature_wrapper_74003input_8"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Ň2Ď
(__inference_conv2d_6_layer_call_fn_74313˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv2d_6_layer_call_and_return_conditional_losses_74324˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Ň2Ď
(__inference_conv2d_7_layer_call_fn_74333˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv2d_7_layer_call_and_return_conditional_losses_74344˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
Ň2Ď
(__inference_conv2d_8_layer_call_fn_74353˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_conv2d_8_layer_call_and_return_conditional_losses_74363˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ž
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Ü2Ů
2__inference_conv2d_transpose_3_layer_call_fn_74372˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷2ô
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_74406˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
Ü2Ů
2__inference_conv2d_transpose_4_layer_call_fn_74415˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷2ô
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_74449˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
˛
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ü2Ů
2__inference_conv2d_transpose_5_layer_call_fn_74458˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷2ô
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_74491˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapperŠ
 __inference__wrapped_model_72824()*+,-./01238˘5
.˘+
)&
input_8˙˙˙˙˙˙˙˙˙
Ş "9Ş6
4
decoder)&
decoder˙˙˙˙˙˙˙˙˙ł
C__inference_conv2d_6_layer_call_and_return_conditional_losses_74324l()7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙ 
 
(__inference_conv2d_6_layer_call_fn_74313_()7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙
Ş " ˙˙˙˙˙˙˙˙˙ ł
C__inference_conv2d_7_layer_call_and_return_conditional_losses_74344l*+7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙ 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙@
 
(__inference_conv2d_7_layer_call_fn_74333_*+7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙ 
Ş " ˙˙˙˙˙˙˙˙˙@ł
C__inference_conv2d_8_layer_call_and_return_conditional_losses_74363l,-7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 
(__inference_conv2d_8_layer_call_fn_74353_,-7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş " ˙˙˙˙˙˙˙˙˙â
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_74406./I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
 ş
2__inference_conv2d_transpose_3_layer_call_fn_74372./I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@â
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_7444901I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
 ş
2__inference_conv2d_transpose_4_layer_call_fn_7441501I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙@
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ â
M__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_7449123I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
Ş "?˘<
52
0+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ş
2__inference_conv2d_transpose_5_layer_call_fn_7445823I˘F
?˘<
:7
inputs+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ 
Ş "2/+˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ż
B__inference_decoder_layer_call_and_return_conditional_losses_73296y./0123@˘=
6˘3
)&
input_7˙˙˙˙˙˙˙˙˙
p 

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ż
B__inference_decoder_layer_call_and_return_conditional_losses_73315y./0123@˘=
6˘3
)&
input_7˙˙˙˙˙˙˙˙˙
p

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ž
B__inference_decoder_layer_call_and_return_conditional_losses_74241x./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ž
B__inference_decoder_layer_call_and_return_conditional_losses_74304x./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 
'__inference_decoder_layer_call_fn_73207l./0123@˘=
6˘3
)&
input_7˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_decoder_layer_call_fn_73277l./0123@˘=
6˘3
)&
input_7˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_decoder_layer_call_fn_74161k./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_decoder_layer_call_fn_74178k./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙ż
B__inference_encoder_layer_call_and_return_conditional_losses_73016y()*+,-@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙
p 

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ż
B__inference_encoder_layer_call_and_return_conditional_losses_73035y()*+,-@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙
p

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ž
B__inference_encoder_layer_call_and_return_conditional_losses_74061x()*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 ž
B__inference_encoder_layer_call_and_return_conditional_losses_74085x()*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş "-˘*
# 
0˙˙˙˙˙˙˙˙˙
 
'__inference_encoder_layer_call_fn_72897l()*+,-@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_encoder_layer_call_fn_72997l()*+,-@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_encoder_layer_call_fn_74020k()*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
'__inference_encoder_layer_call_fn_74037k()*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙ˇ
#__inference_signature_wrapper_74003()*+,-./0123C˘@
˘ 
9Ş6
4
input_8)&
input_8˙˙˙˙˙˙˙˙˙"9Ş6
4
decoder)&
decoder˙˙˙˙˙˙˙˙˙Ă
K__inference_vector_quantizer_layer_call_and_return_conditional_losses_74144t2˘/
(˘%
# 
x˙˙˙˙˙˙˙˙˙
Ş ";˘8
# 
0˙˙˙˙˙˙˙˙˙

	
1/0 
0__inference_vector_quantizer_layer_call_fn_74093Y2˘/
(˘%
# 
x˙˙˙˙˙˙˙˙˙
Ş " ˙˙˙˙˙˙˙˙˙Ô
A__inference_vq_vae_layer_call_and_return_conditional_losses_73611()*+,-./0123@˘=
6˘3
)&
input_8˙˙˙˙˙˙˙˙˙
p 

 
Ş ";˘8
# 
0˙˙˙˙˙˙˙˙˙

	
1/0 Ô
A__inference_vq_vae_layer_call_and_return_conditional_losses_73646()*+,-./0123@˘=
6˘3
)&
input_8˙˙˙˙˙˙˙˙˙
p

 
Ş ";˘8
# 
0˙˙˙˙˙˙˙˙˙

	
1/0 Ó
A__inference_vq_vae_layer_call_and_return_conditional_losses_73840()*+,-./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş ";˘8
# 
0˙˙˙˙˙˙˙˙˙

	
1/0 Ó
A__inference_vq_vae_layer_call_and_return_conditional_losses_73970()*+,-./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş ";˘8
# 
0˙˙˙˙˙˙˙˙˙

	
1/0 
&__inference_vq_vae_layer_call_fn_73436s()*+,-./0123@˘=
6˘3
)&
input_8˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
&__inference_vq_vae_layer_call_fn_73576s()*+,-./0123@˘=
6˘3
)&
input_8˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙
&__inference_vq_vae_layer_call_fn_73678r()*+,-./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p 

 
Ş " ˙˙˙˙˙˙˙˙˙
&__inference_vq_vae_layer_call_fn_73710r()*+,-./0123?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙
p

 
Ş " ˙˙˙˙˙˙˙˙˙