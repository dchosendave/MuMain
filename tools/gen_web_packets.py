#!/usr/bin/env python3
"""Generate the web build's packet builders from OpenMU's packet XML.

The native client sends packets through MUnique.Client.Library, a .NET Native AOT
library whose `Send<Packet>` exports are generated from the same XML. The browser
has no .NET, so the web build implements those exports in C++. The C++ client
loads them by name (LoadManagedSymbol), so nothing outside src/source/Dotnet/web
changes.

Signatures are taken from the checked-in PacketBindings_<Group>.h typedefs (what
the C++ caller passes); the field layout comes from the XML. Both are checked
against each other, so drift between the XML and the bindings fails the build.

The field semantics mirror OpenMU's GenerateRefStructs.xslt and
ByteSpanExtensions.cs (MUnique.OpenMU.Network.Packets).

Usage:
  gen_web_packets.py --xml-dir <contentFiles/any/net10.0> --bindings-dir <src/source/Dotnet>
                     --output <WebPacketBuilders.generated.cpp>
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"pd": "http://www.munique.net/OpenMU/PacketDefinitions"}
GROUPS = [
    ("ClientToServer", "ClientToServer/ClientToServerPackets.xml"),
    ("ChatServer", "ChatServer/ChatServerPackets.xml"),
    ("ConnectServer", "ConnectServer/ConnectServerPackets.xml"),
]
HEADERS = {
    "C1Header": (0xC1, False),
    "C1HeaderWithSubCode": (0xC1, True),
    "C2Header": (0xC2, False),
    "C2HeaderWithSubCode": (0xC2, True),
    "C3Header": (0xC3, False),
    "C3HeaderWithSubCode": (0xC3, True),
    "C4Header": (0xC4, False),
    "C4HeaderWithSubCode": (0xC4, True),
}
INT_TYPES = {
    "ShortLittleEndian": ("uint16_t", "WriteU16Le"),
    "ShortBigEndian": ("uint16_t", "WriteU16Be"),
    "IntegerLittleEndian": ("uint32_t", "WriteU32Le"),
    "IntegerBigEndian": ("uint32_t", "WriteU32Be"),
    "LongLittleEndian": ("uint64_t", "WriteU64Le"),
    "LongBigEndian": ("uint64_t", "WriteU64Be"),
}
TYPEDEF = re.compile(r"typedef void\(CORECLR_DELEGATE_CALLTYPE\* (\w+)\)\((.*)\);")


def clean_name(name):
    return re.sub(r"[ ()/\-:]", "", name)


def lower_first(name):
    result = name[0].lower() + name[1:]
    return "class_" if result == "class" else result


def text(element, tag):
    value = element.find(f"pd:{tag}", NS)
    return value.text.strip() if value is not None and value.text else None


def parse_bindings(path):
    """Returns {function name: [param type, ...]} without the leading handle."""
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = TYPEDEF.match(line.strip())
        if not match:
            continue
        params = [p.strip() for p in match.group(2).split(",")]
        if params[0] != "int32_t":
            raise SystemExit(f"{path.name}: {match.group(1)} does not start with the handle")
        result[match.group(1)] = params[1:]
    return result


class Field:
    def __init__(self, element):
        self.index = int(text(element, "Index"))
        self.type = text(element, "Type")
        self.name = lower_first(clean_name(text(element, "Name")))
        self.length = text(element, "Length")
        self.length = int(self.length) if self.length is not None else None
        self.left_shifted = text(element, "LeftShifted")
        self.left_shifted = int(self.left_shifted) if self.left_shifted is not None else None
        self.default = text(element, "DefaultValue")
        self.type_name = text(element, "TypeName")

    def native_params(self):
        """The parameter types the bindings header declares for this field."""
        if self.type in ("Byte", "Boolean"):
            return ["BYTE"]
        if self.type in INT_TYPES:
            return [INT_TYPES[self.type][0]]
        if self.type == "String":
            return ["const char16_t*"]
        if self.type == "Binary":
            return ["const BYTE*", "uint32_t"]
        if self.type == "Enum":
            return ["uint32_t" if "." in self.type_name else self.type_name]
        raise SystemExit(f"unsupported field type {self.type} ({self.name})")


def emit_packet(group, packet, signature, out):
    name = clean_name(text(packet, "Name"))
    function = f"Send{name}"
    header_type, has_sub_code = HEADERS[text(packet, "HeaderType")]
    code = int(text(packet, "Code"), 16)
    # A sub code header without <SubCode> (ChatServer Authenticate) leaves that byte zero, like the ref struct.
    sub_code = int(text(packet, "SubCode"), 16) if has_sub_code and text(packet, "SubCode") else None
    fixed_length = text(packet, "Length")
    fields = [Field(f) for f in packet.findall("pd:Fields/pd:Field", NS)]

    # Same parameter order as GenerateBindingsHeader.xslt: <xsl:sort select="pd:DefaultValue"/>,
    # a stable string sort where a missing default ("") comes first.
    ordered = sorted(fields, key=lambda f: f.default or "")
    expected = [t for f in ordered for t in f.native_params()]
    if [t.replace(" ", "") for t in signature] != [t.replace(" ", "") for t in expected]:
        raise SystemExit(f"{group}.{function}: bindings {signature} != XML {expected}")

    params = []
    for f in ordered:
        if f.type == "Binary":
            params += [f"const BYTE* {f.name}", f"uint32_t {f.name}ByteLength"]
        else:
            params.append(f"{f.native_params()[0]} {f.name}")

    # The last field is variable when it is a String/Binary without Length. A packet Length still
    # wins (the ref struct is created with it); the field then fills the rest of the packet.
    variable = fields[-1] if fields and fields[-1].type in ("String", "Binary") and fields[-1].length is None else None
    if variable is not None and fixed_length is not None:
        variable.length = int(fixed_length) - variable.index
        variable = None

    out.append(f"// {group}: {text(packet, 'Name')}")
    out.append(f"void {function}(int32_t handle{''.join(', ' + p for p in params)})")
    out.append("{")
    if variable is not None and variable.type == "String":
        out.append(f"    const std::string {variable.name}Utf8 = Utf16ToUtf8({variable.name});")
        size = f"{variable.index} + {variable.name}Utf8.size() + 1"
    elif variable is not None:
        size = f"{variable.index} + static_cast<std::size_t>({variable.name}ByteLength)"
    elif fixed_length is not None:
        size = fixed_length
    else:
        raise SystemExit(f"{group}.{function}: no Length and no variable field")

    out.append(f"    PacketBuffer packet(0x{header_type:02X}, 0x{code:02X}, {'-1' if sub_code is None else f'0x{sub_code:02X}'}, {size});")
    # Fields with a DefaultValue are parameters too, so every field is written from its parameter.
    for f in fields:
        out.extend("    " + line for line in write_field(f, variable))
    out.append("    SubmitPacket(handle, packet);")
    out.append("}")
    out.append("")
    return function


def write_field(f, variable):
    i = f.index
    if f.type == "Boolean":
        # The C# export converts with `== 1`, so only 1 is true.
        return [f"packet.WriteBoolean({i}, {f.name} == 1, {f.left_shifted or 0});"]
    if f.type in ("Byte", "Enum"):
        value = f"static_cast<uint8_t>({f.name})"
        if f.length is not None or f.left_shifted is not None:
            return [f"packet.WriteBits({i}, {value}, {f.length if f.length is not None else 8}, {f.left_shifted or 0});"]
        return [f"packet.WriteByte({i}, {value});"]
    if f.type in INT_TYPES:
        return [f"packet.{INT_TYPES[f.type][1]}({i}, {f.name});"]
    if f.type == "String":
        if f is variable:
            return [f"packet.WriteBytes({i}, {f.name}Utf8.data(), {f.name}Utf8.size());"]
        return [f"if (!packet.WriteFixedString({i}, {f.length}, {f.name})) return;"]
    if f.type == "Binary":
        if f is variable:
            return [f"packet.WriteBytes({i}, {f.name}, {f.name}ByteLength);"]
        return [f"if (!packet.WriteFixedBinary({i}, {f.length}, {f.name}, {f.name}ByteLength)) return;"]
    raise SystemExit(f"unsupported field type {f.type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", required=True, type=Path)
    parser.add_argument("--bindings-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    out = [
        "// Generated by tools/gen_web_packets.py from OpenMU's packet XML. Do not edit.",
        "// Web build implementation of MUnique.Client.Library's Send* exports.",
        "",
        '#include "stdafx.h"',
        "",
        '#include "Dotnet/PacketFunctions_CommonEnums.h"',
    ]
    out += [f'#include "Dotnet/PacketFunctions_{group}_Enums.h"' for group, _ in GROUPS]
    out += [
        '#include "Dotnet/web/WebPacketBuffer.h"',
        "",
        "namespace mu::web::packets",
        "{",
        "",
    ]

    functions = []
    for group, relative in GROUPS:
        bindings = parse_bindings(args.bindings_dir / f"PacketBindings_{group}.h")
        root = ET.parse(args.xml_dir / relative).getroot()
        seen = set()
        for packet in root.findall("pd:Packets/pd:Packet", NS):
            name = "Send" + clean_name(text(packet, "Name"))
            if name not in bindings:
                continue  # server-to-client packets and Structure[] packets have no binding
            seen.add(name)
            functions.append(emit_packet(group, packet, bindings[name], out))
        missing = set(bindings) - seen
        if missing:
            raise SystemExit(f"{group}: bindings without XML packet: {sorted(missing)}")

    out += [
        "} // namespace mu::web::packets",
        "",
        "namespace mu::web",
        "{",
        "const NetworkSymbol kGeneratedPacketSymbols[] = {",
    ]
    out += [f'    {{"{f}", reinterpret_cast<void*>(&packets::{f})}},' for f in functions]
    out += [
        "};",
        "const std::size_t kGeneratedPacketSymbolCount = std::size(kGeneratedPacketSymbols);",
        "} // namespace mu::web",
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(out)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
        args.output.write_text(content, encoding="utf-8")
    print(f"gen_web_packets: {len(functions)} packet builders -> {args.output}")


if __name__ == "__main__":
    sys.exit(main())
