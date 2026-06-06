import os
from contextlib import AsyncExitStack
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

mcp_router = APIRouter(prefix="/api/mcp", tags=["MCP Agent Builder Bridge"])

class MCPContext:
    session: ClientSession = None
    exit_stack: AsyncExitStack = None

mcp_ctx = MCPContext()

async def start_mcp():
    """Starts the official MongoDB MCP Server as a subprocess and bridges it."""
    mcp_ctx.exit_stack = AsyncExitStack()
    
    # Use globally installed package (baked into Docker image) for zero cold-start.
    # Falls back to npx for local development where npm install -g wasn't run.
    import shutil
    mcp_bin = shutil.which("mongodb-mcp-server")  # installed globally by npm install -g
    if mcp_bin:
        command, args = "node", [mcp_bin]
    else:
        command, args = "npx", ["-y", "@mongodb-js/mongodb-mcp-server"]

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env={
            "MONGODB_URI": os.getenv("MONGODB_URI", ""),
            "MDB_MCP_CONNECTION_STRING": os.getenv("MONGODB_URI", ""),
            "PATH": os.environ.get("PATH", "")
        }
    )
    
    try:
        transport = await mcp_ctx.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = transport
        
        session = await mcp_ctx.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        
        # Explicitly connect to the database to prevent intermittent offline errors
        mongo_uri = os.getenv("MONGODB_URI", "")
        if mongo_uri:
            try:
                await session.call_tool("connect", arguments={"connectionStringOrClusterName": mongo_uri})
                print("[MCP Bridge] Explicitly connected to MongoDB Atlas via 'connect' tool.")
            except Exception as conn_err:
                print(f"[MCP Bridge] Failed to run 'connect' tool: {conn_err}")

        mcp_ctx.session = session
        print("[MCP Bridge] Successfully connected to official MongoDB MCP Server!")
    except Exception as e:
        print(f"[MCP Bridge Error] Could not start MCP Server: {e}")

async def stop_mcp():
    if mcp_ctx.exit_stack:
        await mcp_ctx.exit_stack.aclose()

@mcp_router.get("/tools")
async def list_mcp_tools():
    """
    Google Agent Builder calls this to dynamically discover what tools MongoDB Atlas provides natively.
    """
    if not mcp_ctx.session:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    response = await mcp_ctx.session.list_tools()
    tools = []
    for t in response.tools:
        tools.append({
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema
        })
    return {"tools": tools}

class ExecuteRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]

@mcp_router.post("/execute")
async def execute_mcp_tool(req: ExecuteRequest):
    """
    Google Agent Builder uses this OpenAPI REST endpoint to securely execute MCP JSON-RPC tools on MongoDB.
    """
    if not mcp_ctx.session:
        raise HTTPException(status_code=503, detail="MCP Server not initialized")
    
    try:
        result = await mcp_ctx.session.call_tool(req.tool_name, arguments=req.arguments)
        
        out = []
        for c in result.content:
            if c.type == "text":
                out.append(c.text)
        return {"result": "\n".join(out), "isError": result.isError}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
