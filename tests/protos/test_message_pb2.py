# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: test_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='test_message.proto',
  package='mlflow',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n\x12test_message.proto\x12\x06mlflow\"\x9d\t\n\x0bTestMessage\x12\x13\n\x0b\x66ield_int32\x18\x01 \x01(\x05\x12\x13\n\x0b\x66ield_int64\x18\x02 \x01(\x03\x12\x14\n\x0c\x66ield_uint32\x18\x03 \x01(\r\x12\x14\n\x0c\x66ield_uint64\x18\x04 \x01(\x04\x12\x14\n\x0c\x66ield_sint32\x18\x05 \x01(\x11\x12\x14\n\x0c\x66ield_sint64\x18\x06 \x01(\x12\x12\x15\n\rfield_fixed32\x18\x07 \x01(\x07\x12\x15\n\rfield_fixed64\x18\x08 \x01(\x06\x12\x16\n\x0e\x66ield_sfixed32\x18\t \x01(\x0f\x12\x16\n\x0e\x66ield_sfixed64\x18\n \x01(\x10\x12\x12\n\nfield_bool\x18\x0b \x01(\x08\x12\x14\n\x0c\x66ield_string\x18\x0c \x01(\t\x12 \n\x13\x66ield_with_default1\x18\r \x01(\x03:\x03\x31\x30\x30\x12 \n\x13\x66ield_with_default2\x18\x0e \x01(\x03:\x03\x32\x30\x30\x12\x1c\n\x14\x66ield_repeated_int64\x18\x0f \x03(\x03\x12\x30\n\nfield_enum\x18\x10 \x01(\x0e\x32\x1c.mlflow.TestMessage.TestEnum\x12\x41\n\x13\x66ield_inner_message\x18\x11 \x03(\x0b\x32$.mlflow.TestMessage.TestInnerMessage\x12\x10\n\x06oneof1\x18\x12 \x01(\x03H\x00\x12\x10\n\x06oneof2\x18\x13 \x01(\x03H\x00\x12\x36\n\nfield_map1\x18\x14 \x03(\x0b\x32\".mlflow.TestMessage.FieldMap1Entry\x12\x36\n\nfield_map2\x18\x15 \x03(\x0b\x32\".mlflow.TestMessage.FieldMap2Entry\x12\x36\n\nfield_map3\x18\x16 \x03(\x0b\x32\".mlflow.TestMessage.FieldMap3Entry\x12\x36\n\nfield_map4\x18\x17 \x03(\x0b\x32\".mlflow.TestMessage.FieldMap4Entry\x1am\n\x10TestInnerMessage\x12\x19\n\x11\x66ield_inner_int64\x18\x01 \x01(\x03\x12\"\n\x1a\x66ield_inner_repeated_int64\x18\x02 \x03(\x03\x12\x1a\n\x12\x66ield_inner_string\x18\x03 \x01(\t\x1a\x30\n\x0e\x46ieldMap1Entry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1a\x30\n\x0e\x46ieldMap2Entry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1a\x30\n\x0e\x46ieldMap3Entry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1aV\n\x0e\x46ieldMap4Entry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\x33\n\x05value\x18\x02 \x01(\x0b\x32$.mlflow.TestMessage.TestInnerMessage:\x02\x38\x01\"6\n\x08TestEnum\x12\x08\n\x04NONE\x10\x00\x12\x0f\n\x0b\x45NUM_VALUE1\x10\x01\x12\x0f\n\x0b\x45NUM_VALUE2\x10\x02*\x06\x08\xe8\x07\x10\xd0\x0f\x42\x0c\n\ntest_oneof\"F\n\x10\x45xtensionMessage22\n\x14\x66ield_extended_int64\x12\x13.mlflow.TestMessage\x18\xe9\x07 \x01(\x03')
)



_TESTMESSAGE_TESTENUM = _descriptor.EnumDescriptor(
  name='TestEnum',
  full_name='mlflow.TestMessage.TestEnum',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ENUM_VALUE1', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ENUM_VALUE2', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1136,
  serialized_end=1190,
)
_sym_db.RegisterEnumDescriptor(_TESTMESSAGE_TESTENUM)


_TESTMESSAGE_TESTINNERMESSAGE = _descriptor.Descriptor(
  name='TestInnerMessage',
  full_name='mlflow.TestMessage.TestInnerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='field_inner_int64', full_name='mlflow.TestMessage.TestInnerMessage.field_inner_int64', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_inner_repeated_int64', full_name='mlflow.TestMessage.TestInnerMessage.field_inner_repeated_int64', index=1,
      number=2, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_inner_string', full_name='mlflow.TestMessage.TestInnerMessage.field_inner_string', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=787,
  serialized_end=896,
)

_TESTMESSAGE_FIELDMAP1ENTRY = _descriptor.Descriptor(
  name='FieldMap1Entry',
  full_name='mlflow.TestMessage.FieldMap1Entry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='mlflow.TestMessage.FieldMap1Entry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='mlflow.TestMessage.FieldMap1Entry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=898,
  serialized_end=946,
)

_TESTMESSAGE_FIELDMAP2ENTRY = _descriptor.Descriptor(
  name='FieldMap2Entry',
  full_name='mlflow.TestMessage.FieldMap2Entry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='mlflow.TestMessage.FieldMap2Entry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='mlflow.TestMessage.FieldMap2Entry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=948,
  serialized_end=996,
)

_TESTMESSAGE_FIELDMAP3ENTRY = _descriptor.Descriptor(
  name='FieldMap3Entry',
  full_name='mlflow.TestMessage.FieldMap3Entry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='mlflow.TestMessage.FieldMap3Entry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='mlflow.TestMessage.FieldMap3Entry.value', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=998,
  serialized_end=1046,
)

_TESTMESSAGE_FIELDMAP4ENTRY = _descriptor.Descriptor(
  name='FieldMap4Entry',
  full_name='mlflow.TestMessage.FieldMap4Entry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='mlflow.TestMessage.FieldMap4Entry.key', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='mlflow.TestMessage.FieldMap4Entry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1048,
  serialized_end=1134,
)

_TESTMESSAGE = _descriptor.Descriptor(
  name='TestMessage',
  full_name='mlflow.TestMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='field_int32', full_name='mlflow.TestMessage.field_int32', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_int64', full_name='mlflow.TestMessage.field_int64', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_uint32', full_name='mlflow.TestMessage.field_uint32', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_uint64', full_name='mlflow.TestMessage.field_uint64', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_sint32', full_name='mlflow.TestMessage.field_sint32', index=4,
      number=5, type=17, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_sint64', full_name='mlflow.TestMessage.field_sint64', index=5,
      number=6, type=18, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_fixed32', full_name='mlflow.TestMessage.field_fixed32', index=6,
      number=7, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_fixed64', full_name='mlflow.TestMessage.field_fixed64', index=7,
      number=8, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_sfixed32', full_name='mlflow.TestMessage.field_sfixed32', index=8,
      number=9, type=15, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_sfixed64', full_name='mlflow.TestMessage.field_sfixed64', index=9,
      number=10, type=16, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_bool', full_name='mlflow.TestMessage.field_bool', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_string', full_name='mlflow.TestMessage.field_string', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_with_default1', full_name='mlflow.TestMessage.field_with_default1', index=12,
      number=13, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=100,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_with_default2', full_name='mlflow.TestMessage.field_with_default2', index=13,
      number=14, type=3, cpp_type=2, label=1,
      has_default_value=True, default_value=200,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_repeated_int64', full_name='mlflow.TestMessage.field_repeated_int64', index=14,
      number=15, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_enum', full_name='mlflow.TestMessage.field_enum', index=15,
      number=16, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_inner_message', full_name='mlflow.TestMessage.field_inner_message', index=16,
      number=17, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='oneof1', full_name='mlflow.TestMessage.oneof1', index=17,
      number=18, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='oneof2', full_name='mlflow.TestMessage.oneof2', index=18,
      number=19, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_map1', full_name='mlflow.TestMessage.field_map1', index=19,
      number=20, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_map2', full_name='mlflow.TestMessage.field_map2', index=20,
      number=21, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_map3', full_name='mlflow.TestMessage.field_map3', index=21,
      number=22, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='field_map4', full_name='mlflow.TestMessage.field_map4', index=22,
      number=23, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TESTMESSAGE_TESTINNERMESSAGE, _TESTMESSAGE_FIELDMAP1ENTRY, _TESTMESSAGE_FIELDMAP2ENTRY, _TESTMESSAGE_FIELDMAP3ENTRY, _TESTMESSAGE_FIELDMAP4ENTRY, ],
  enum_types=[
    _TESTMESSAGE_TESTENUM,
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 2000), ],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='test_oneof', full_name='mlflow.TestMessage.test_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=31,
  serialized_end=1212,
)


_EXTENSIONMESSAGE = _descriptor.Descriptor(
  name='ExtensionMessage',
  full_name='mlflow.ExtensionMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
    _descriptor.FieldDescriptor(
      name='field_extended_int64', full_name='mlflow.ExtensionMessage.field_extended_int64', index=0,
      number=1001, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=True, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1214,
  serialized_end=1284,
)

_TESTMESSAGE_TESTINNERMESSAGE.containing_type = _TESTMESSAGE
_TESTMESSAGE_FIELDMAP1ENTRY.containing_type = _TESTMESSAGE
_TESTMESSAGE_FIELDMAP2ENTRY.containing_type = _TESTMESSAGE
_TESTMESSAGE_FIELDMAP3ENTRY.containing_type = _TESTMESSAGE
_TESTMESSAGE_FIELDMAP4ENTRY.fields_by_name['value'].message_type = _TESTMESSAGE_TESTINNERMESSAGE
_TESTMESSAGE_FIELDMAP4ENTRY.containing_type = _TESTMESSAGE
_TESTMESSAGE.fields_by_name['field_enum'].enum_type = _TESTMESSAGE_TESTENUM
_TESTMESSAGE.fields_by_name['field_inner_message'].message_type = _TESTMESSAGE_TESTINNERMESSAGE
_TESTMESSAGE.fields_by_name['field_map1'].message_type = _TESTMESSAGE_FIELDMAP1ENTRY
_TESTMESSAGE.fields_by_name['field_map2'].message_type = _TESTMESSAGE_FIELDMAP2ENTRY
_TESTMESSAGE.fields_by_name['field_map3'].message_type = _TESTMESSAGE_FIELDMAP3ENTRY
_TESTMESSAGE.fields_by_name['field_map4'].message_type = _TESTMESSAGE_FIELDMAP4ENTRY
_TESTMESSAGE_TESTENUM.containing_type = _TESTMESSAGE
_TESTMESSAGE.oneofs_by_name['test_oneof'].fields.append(
  _TESTMESSAGE.fields_by_name['oneof1'])
_TESTMESSAGE.fields_by_name['oneof1'].containing_oneof = _TESTMESSAGE.oneofs_by_name['test_oneof']
_TESTMESSAGE.oneofs_by_name['test_oneof'].fields.append(
  _TESTMESSAGE.fields_by_name['oneof2'])
_TESTMESSAGE.fields_by_name['oneof2'].containing_oneof = _TESTMESSAGE.oneofs_by_name['test_oneof']
DESCRIPTOR.message_types_by_name['TestMessage'] = _TESTMESSAGE
DESCRIPTOR.message_types_by_name['ExtensionMessage'] = _EXTENSIONMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TestMessage = _reflection.GeneratedProtocolMessageType('TestMessage', (_message.Message,), dict(

  TestInnerMessage = _reflection.GeneratedProtocolMessageType('TestInnerMessage', (_message.Message,), dict(
    DESCRIPTOR = _TESTMESSAGE_TESTINNERMESSAGE,
    __module__ = 'test_message_pb2'
    # @@protoc_insertion_point(class_scope:mlflow.TestMessage.TestInnerMessage)
    ))
  ,

  FieldMap1Entry = _reflection.GeneratedProtocolMessageType('FieldMap1Entry', (_message.Message,), dict(
    DESCRIPTOR = _TESTMESSAGE_FIELDMAP1ENTRY,
    __module__ = 'test_message_pb2'
    # @@protoc_insertion_point(class_scope:mlflow.TestMessage.FieldMap1Entry)
    ))
  ,

  FieldMap2Entry = _reflection.GeneratedProtocolMessageType('FieldMap2Entry', (_message.Message,), dict(
    DESCRIPTOR = _TESTMESSAGE_FIELDMAP2ENTRY,
    __module__ = 'test_message_pb2'
    # @@protoc_insertion_point(class_scope:mlflow.TestMessage.FieldMap2Entry)
    ))
  ,

  FieldMap3Entry = _reflection.GeneratedProtocolMessageType('FieldMap3Entry', (_message.Message,), dict(
    DESCRIPTOR = _TESTMESSAGE_FIELDMAP3ENTRY,
    __module__ = 'test_message_pb2'
    # @@protoc_insertion_point(class_scope:mlflow.TestMessage.FieldMap3Entry)
    ))
  ,

  FieldMap4Entry = _reflection.GeneratedProtocolMessageType('FieldMap4Entry', (_message.Message,), dict(
    DESCRIPTOR = _TESTMESSAGE_FIELDMAP4ENTRY,
    __module__ = 'test_message_pb2'
    # @@protoc_insertion_point(class_scope:mlflow.TestMessage.FieldMap4Entry)
    ))
  ,
  DESCRIPTOR = _TESTMESSAGE,
  __module__ = 'test_message_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.TestMessage)
  ))
_sym_db.RegisterMessage(TestMessage)
_sym_db.RegisterMessage(TestMessage.TestInnerMessage)
_sym_db.RegisterMessage(TestMessage.FieldMap1Entry)
_sym_db.RegisterMessage(TestMessage.FieldMap2Entry)
_sym_db.RegisterMessage(TestMessage.FieldMap3Entry)
_sym_db.RegisterMessage(TestMessage.FieldMap4Entry)

ExtensionMessage = _reflection.GeneratedProtocolMessageType('ExtensionMessage', (_message.Message,), dict(
  DESCRIPTOR = _EXTENSIONMESSAGE,
  __module__ = 'test_message_pb2'
  # @@protoc_insertion_point(class_scope:mlflow.ExtensionMessage)
  ))
_sym_db.RegisterMessage(ExtensionMessage)

TestMessage.RegisterExtension(_EXTENSIONMESSAGE.extensions_by_name['field_extended_int64'])

_TESTMESSAGE_FIELDMAP1ENTRY._options = None
_TESTMESSAGE_FIELDMAP2ENTRY._options = None
_TESTMESSAGE_FIELDMAP3ENTRY._options = None
_TESTMESSAGE_FIELDMAP4ENTRY._options = None
# @@protoc_insertion_point(module_scope)