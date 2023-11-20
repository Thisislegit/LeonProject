#ifndef ML_UTIL_H
#define ML_UTIL_H
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "utils/ruleutils.h"


// JSON tags for sending to the leon server.
static const char* START_QUERY_MESSAGE = "{\"type\": \"query\"}\n";
static const char* START_SHOULD_OPT_MESSAGE = "{\"type\": \"should_opt\"}\n";
static const char *START_FEEDBACK_MESSAGE = "{\"type\": \"reward\"}\n";
static const char* START_PREDICTION_MESSAGE = "{\"type\": \"predict\"}\n";
static const char* TERMINAL_MESSAGE = "{\"final\": true}\n";

typedef struct LeonState
{
    MemoryContext leonContext;
    // other state-related fields
	char *leon_host;
	int leon_port;
} LeonState;



static int 
custom_snprintf(char **buf, size_t *buf_size, const char *format, ...) {
    va_list args;
    int required, current_length, total_required;

    current_length = strlen(*buf);
    va_start(args, format);
    required = vsnprintf(NULL, 0, format, args); // Get required space
    va_end(args);

    total_required = current_length + required + 1; // +1 for null terminator

    if (total_required > *buf_size) {
        *buf_size = total_required;
        *buf = (char *) repalloc(*buf, *buf_size);
    }

    va_start(args, format);
    vsnprintf(*buf + current_length, *buf_size - current_length, format, args);
    va_end(args);

    return total_required - 1; // Return total characters in buffer excluding null terminator
}


typedef struct
{
	List	   *rtable;			/* List of RangeTblEntry nodes */
	List	   *rtable_names;	/* Parallel list of names for RTEs */
	List	   *rtable_columns; /* Parallel list of deparse_columns structs */
	List	   *subplans;		/* List of Plan trees for SubPlans */
	List	   *ctes;			/* List of CommonTableExpr nodes */
	AppendRelInfo **appendrels; /* Array of AppendRelInfo nodes, or NULL */
	/* Workspace for column alias assignment: */
	bool		unique_using;	/* Are we making USING names globally unique */
	List	   *using_names;	/* List of assigned names for USING columns */
	/* Remaining fields are used only when deparsing a Plan tree: */
	Plan	   *plan;			/* immediate parent of current expression */
	List	   *ancestors;		/* ancestors of plan */
	Plan	   *outer_plan;		/* outer subnode, or NULL if none */
	Plan	   *inner_plan;		/* inner subnode, or NULL if none */
	List	   *outer_tlist;	/* referent for OUTER_VAR Vars */
	List	   *inner_tlist;	/* referent for INNER_VAR Vars */
	List	   *index_tlist;	/* referent for INDEX_VAR Vars */
	/* Special namespace representing a function signature: */
	char	   *funcname;
	int			numargs;
	char	  **argnames;
} deparse_namespace;
extern void set_simple_column_names(deparse_namespace *dpns);
extern char *deparse_expression_pretty(Node *expr, List *dpcontext,
									   bool forceprefix, bool showimplicit,
									   int prettyFlags, int startIndent);


static void get_calibrations(double calibrations[], uint32 queryid, int32_t length, int conn_fd, int* picknode_index){
  		// Read the response from the server and store it in the calibrations array
      // one element is like "1.12," length 5
	  // one element is like "1.12,0,6;"
	  char *response = (char *)palloc0(9 * length * sizeof(char));
	  char *buffer = (char *)palloc0(9 * length * sizeof(char));
	  bool read_flag = false;
	  ssize_t need = 0;
	  ssize_t bytesRead;
	  while(need != 9 * length * sizeof(char))
	  {
		bytesRead = read(conn_fd, buffer, 9 * length * sizeof(char));
		if (bytesRead != -1)
		{
			memcpy(response + need, buffer, bytesRead);
			need += bytesRead;
			// elog(WARNING, "%zd", bytesRead);
			// elog(WARNING, "%zd", need);
		}
			read_flag = true;
	  }
	  pfree(buffer);
      if (read_flag) 
      {
		char* unit;
		char* token;
		char* restUnit;
		char* restToken;
		int i = 0;

        unit = strtok_r(response, ";", &restUnit);
    	while (unit != NULL && i < length) {
        // Split each unit into components using comma
        token = strtok_r(unit, ",", &restToken);
        if (token == NULL) break;
        double mantissa = atof(token); // Get mantissa

        token = strtok_r(NULL, ",", &restToken);
        if (token == NULL) break;
        int exponentSignValue = atoi(token); // Get sign of exponent

        token = strtok_r(NULL, ",", &restToken);
        if (token == NULL) break;
        int exponent = atoi(token); // Get exponent

		// Determine actual sign of exponent
        int actualExponent = exponentSignValue == 0 ? -exponent : exponent;

		// Find the plan index that we pick
		if (mantissa == 0.01 && actualExponent == -9)
        {
            *picknode_index = i;
        }

        // Calculate the actual value
        calibrations[i] = mantissa * pow(10, actualExponent);

        // Move to the next unit for the next iteration
        unit = strtok_r(NULL, ";", &restUnit);
        i++; // Increment index after processing a unit
    	}

        Assert(i == length);
        if (i != length)
        {
          elog(ERROR, "Python code get wrong number of results!");
          exit(1);
        }
      } else {
        shutdown(conn_fd, SHUT_RDWR);
		if (conn_fd > 0) close(conn_fd);
        elog(WARNING, "LEON could not read the response from the server.");
      }
    
      pfree(response);
}


static int connect_to_leon(const char* host, int port) {
  int ret, conn_fd;
  struct sockaddr_in server_addr = { 0 };

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port);
  inet_pton(AF_INET, host, &server_addr.sin_addr);
  conn_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (conn_fd < 0) {
    return conn_fd;
  }

	int opt = 1;
  // 设置SO_REUSEADDR选项
	if (setsockopt(conn_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) == -1) {
		elog(WARNING, "setsockopt failed");
		close(conn_fd);
		exit(1);
	}

  
  ret = connect(conn_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
  if (ret == -1) {
    return ret;
  }

  return conn_fd;

}


static void write_all_to_socket(int conn_fd, const char* json) {
  size_t json_length;
  ssize_t written, written_total;
  json_length = strlen(json);
  written_total = 0;
  
  while (written_total != json_length) {
    written = write(conn_fd,
                    json + written_total,
                    json_length - written_total);
    written_total += written;
  }
}

List *
deparse_context_for_path(PlannerInfo *root, List *rtable_names)
{
	deparse_namespace *dpns;

	dpns = (deparse_namespace *) palloc0(sizeof(deparse_namespace));

	/* Initialize fields that stay the same across the whole plan tree */
	dpns->rtable = root->parse->rtable; 
	dpns->rtable_names = rtable_names;
	dpns->subplans = root->glob->subplans;
	dpns->ctes = NIL;
	if (root->glob->appendRelations)
	{
		/* Set up the array, indexed by child relid */
		int			ntables = list_length(dpns->rtable);
		ListCell   *lc;

		dpns->appendrels = (AppendRelInfo **)
			palloc0((ntables + 1) * sizeof(AppendRelInfo *));
		foreach(lc, root->glob->appendRelations)
		{
			AppendRelInfo *appinfo = lfirst_node(AppendRelInfo, lc);
			Index		crelid = appinfo->child_relid;

			Assert(crelid > 0 && crelid <= ntables);
			Assert(dpns->appendrels[crelid] == NULL);
			dpns->appendrels[crelid] = appinfo;
		}
	}
	else
		dpns->appendrels = NULL;	/* don't need it */

	/*
	 * Set up column name aliases.  We will get rather bogus results for join
	 * RTEs, but that doesn't matter because plan trees don't contain any join
	 * alias Vars.
	 */
	set_simple_column_names(dpns);

	/* Return a one-deep namespace stack */
	return list_make1(dpns);
}

static void
debug_print_relids(PlannerInfo *root, Relids relids, char **buf, size_t *buf_size)
{
	int			x;
	bool		first = true;

	x = -1;
	while ((x = bms_next_member(relids, x)) >= 0)
	{
		if (!first)
			custom_snprintf(buf, buf_size, " ");
		if (x < root->simple_rel_array_size &&
			root->simple_rte_array[x])
			custom_snprintf(buf, buf_size, "%s", root->simple_rte_array[x]->eref->aliasname);
		else
			custom_snprintf(buf, buf_size, "%d", x);
		first = false;
	}
}

void
debug_print_joincond(PlannerInfo *root, RelOptInfo *rel, char **stream, size_t *buf_size)
{
	ListCell   *lc;
	List *rtable = root->parse->rtable;

	if (rel->reloptkind != RELOPT_JOINREL)
		return;

	bool first = true;
	foreach(lc, root->parse->jointree->quals)
	{
		Node *expr = (Node *) lfirst(lc);
		if IsA(expr, OpExpr)
		{	
			const OpExpr *e = (const OpExpr *) expr;
			char	   *opname;

			opname = get_opname(e->opno);
			if (list_length(e->args) > 1)
			{	
				Node * left_node = get_leftop((const Expr *) e);
				Node * right_node = get_rightop((const Expr *) e);
				if (IsA(left_node, Var) && IsA(right_node, Var))
				{	
					Var *left_var = (Var *) left_node;
					Var *right_var = (Var *) right_node;
					//Both vars are from the same relation
					if (bms_is_member(left_var->varno, rel->relids) &&
						bms_is_member(right_var->varno, rel->relids))
					{	
						if (!first)
							custom_snprintf(stream, buf_size, ", ");
						debug_print_expr(left_node, rtable, stream, buf_size);
						custom_snprintf(stream, buf_size, " %s ", ((opname != NULL) ? opname : "(invalid operator)"));
						debug_print_expr(right_node, rtable, stream, buf_size);
						first = false;
					}
				}
			}		
		}
	}
}

/*
 * debug_print_expr
 *	  print an expression to a file
 */
void
debug_print_expr(const Node *expr, const List *rtable, char **stream, size_t *buf_size)
{
	if (expr == NULL)
	{
		custom_snprintf(stream, buf_size, "<>");
		return;
	}

	if (IsA(expr, Var))
	{
		const Var  *var = (const Var *) expr;
		char	   *relname,
				   *attname;

		switch (var->varno)
		{
			case INNER_VAR:
				relname = "INNER";
				attname = "?";
				break;
			case OUTER_VAR:
				relname = "OUTER";
				attname = "?";
				break;
			case INDEX_VAR:
				relname = "INDEX";
				attname = "?";
				break;
			default:
				{
					RangeTblEntry *rte;

					Assert(var->varno > 0 &&
						   (int) var->varno <= list_length(rtable));
					rte = rt_fetch(var->varno, rtable);
					relname = rte->eref->aliasname;
					attname = get_rte_attribute_name(rte, var->varattno);
				}
				break;
		}
		custom_snprintf(stream, buf_size, "%s.%s", relname, attname);
	}
	else if (IsA(expr, Const))
	{
		const Const *c = (const Const *) expr;
		Oid			typoutput;
		bool		typIsVarlena;
		char	   *outputstr;

		if (c->constisnull)
		{
			custom_snprintf(stream, buf_size, "NULL");
			return;
		}

		getTypeOutputInfo(c->consttype,
						  &typoutput, &typIsVarlena);

		outputstr = OidOutputFunctionCall(typoutput, c->constvalue);
		// custom_snprintf(stream, buf_size, "\'%s\'", outputstr);
		char *outputstr_escaped = palloc0(strlen(outputstr) + 3);
		sprintf(outputstr_escaped, "\'%s\'", outputstr);
		debug_escape_json(stream, buf_size, outputstr_escaped);
		pfree(outputstr_escaped);
		pfree(outputstr);
	}
	else if (IsA(expr, OpExpr))
	{
		const OpExpr *e = (const OpExpr *) expr;
		char	   *opname;

		opname = get_opname(e->opno);
		if (list_length(e->args) > 1)
		{
			debug_print_expr(get_leftop((const Expr *) e), rtable,  stream, buf_size);
			custom_snprintf(stream, buf_size, " %s ", ((opname != NULL) ? opname : "(invalid operator)"));
			debug_print_expr(get_rightop((const Expr *) e), rtable, stream, buf_size);
		}
		else
		{
			custom_snprintf(stream, buf_size, "%s ", ((opname != NULL) ? opname : "(invalid operator)"));
			debug_print_expr(get_leftop((const Expr *) e), rtable, stream, buf_size);
		}
	}
	else if (IsA(expr, FuncExpr))
	{
		const FuncExpr *e = (const FuncExpr *) expr;
		char	   *funcname;
		ListCell   *l;

		funcname = get_func_name(e->funcid);
		custom_snprintf(stream, buf_size, "%s(", ((funcname != NULL) ? funcname : "(invalid function)"));
		foreach(l, e->args)
		{
			debug_print_expr(lfirst(l), rtable, stream, buf_size);
			if (lnext(e->args, l))
				custom_snprintf(stream, buf_size, ",");
		}
		custom_snprintf(stream, buf_size, ")");
	}
	else if (IsA(expr, RelabelType))
 	{
		const RelabelType *r = (const RelabelType*) expr;

		debug_print_expr((Node *) r->arg, rtable, stream, buf_size);
	}
	else if (IsA(expr, RangeTblRef))
	{
		int	varno = ((RangeTblRef *) expr)->rtindex;
		RangeTblEntry *rte = rt_fetch(varno, rtable);
		custom_snprintf(stream, buf_size, "RTE %d (%s)", varno, rte->eref->aliasname);
	}
	else
		custom_snprintf(stream, buf_size, "unknown expr");
}

List *
create_context(PlannerInfo *root)
{
	List *rtable_names = NIL;
	ListCell *lc;
	foreach(lc, root->parse->rtable)
	{
		RangeTblEntry *rte = lfirst(lc);
		rtable_names = lappend(rtable_names, rte->eref->aliasname);
	}
	List * context = deparse_context_for_path(root, rtable_names);
	return context;
}

void 
delete_context(List *context)
{
	ListCell *lc;
	foreach(lc, context)
	{
		deparse_namespace *dpns = lfirst(lc);
		pfree(dpns);
	}
	list_free(context);
}


/*
 * Produce a JSON string literal, properly escaping characters in the text.
 */
void
debug_escape_json(char **stream, size_t *buf_size, const char *str)
{
	const char *p;

    for (p = str; *p; p++)
	{
		switch (*p)
		{
			case '\b':
                custom_snprintf(stream, buf_size, "\\b");
				break;
			case '\f':
				custom_snprintf(stream, buf_size, "\\f");
				break;
			case '\n':
				custom_snprintf(stream, buf_size, "\\n");
				break;
			case '\r':
				custom_snprintf(stream, buf_size, "\\r");
				break;
			case '\t':
				custom_snprintf(stream, buf_size, "\\t");
				break;
			case '"':
				custom_snprintf(stream, buf_size, "\\\"");
				break;
			case '\\':
				custom_snprintf(stream, buf_size, "\\\\");
				break;
			default:
				if ((unsigned char) *p < ' ')
                    custom_snprintf(stream, buf_size, "\\u%04x", (int) *p);
				else
                    custom_snprintf(stream, buf_size, "%c", *p);
				break;
        }
    }
}

static void
debug_print_restrictclauses(PlannerInfo *root, List *clauses, List *context, char **stream, size_t *buf_size)
{
	ListCell   *l;
	foreach(l, clauses)
	{
		RestrictInfo *c = lfirst(l);
		// char * str = deparse_expression(c->clause, context, true, false);
		char * str = deparse_expression_pretty(c->clause, context, true,
									 false, 0, 0);
        debug_escape_json(stream, buf_size, str);
		// custom_snprintf(stream, buf_size, "%s", str);
		if (str)
			pfree(str);
		// pfree context
		if (lnext(clauses, l))
			custom_snprintf(stream, buf_size, ", ");
	}
}

static void
debug_print_path(PlannerInfo *root, Path *path, int indent, char **stream, size_t *buf_size)
{
	const char *ptype;
	bool join = false;
	Path *subpath = NULL;
	int i;
	// StringInfoData buf;
	char *pathBufPtr = NULL;

	// initStringInfo(&buf);

	switch (nodeTag(path))
	{
		case T_Path:
			switch (path->pathtype)
			{
				case T_SeqScan:
					ptype = "SeqScan";
					break;
				case T_SampleScan:
					ptype = "SampleScan";
					break;
				case T_FunctionScan:
					ptype = "FunctionScan";
					break;
				case T_TableFuncScan:
					ptype = "TableFuncScan";
					break;
				case T_ValuesScan:
					ptype = "ValuesScan";
					break;
				case T_CteScan:
					ptype = "CteScan";
					break;
				case T_NamedTuplestoreScan:
					ptype = "NamedTuplestoreScan";
					break;
				case T_Result:
					ptype = "Result";
					break;
				case T_WorkTableScan:
					ptype = "WorkTableScan";
					break;
				default:
					ptype = "???Path";
					break;
			}
			break;
		case T_IndexPath:
			ptype = "IndexScan";
			break;
		case T_BitmapHeapPath:
			ptype = "BitmapHeapScan";
			break;
		case T_BitmapAndPath:
			ptype = "BitmapAndPath";
			break;
		case T_BitmapOrPath:
			ptype = "BitmapOrPath";
			break;
		case T_TidPath:
			ptype = "TidScan";
			break;
		case T_SubqueryScanPath:
			ptype = "SubqueryScan";
			break;
		case T_ForeignPath:
			ptype = "ForeignScan";
			break;
		case T_CustomPath:
			ptype = "CustomScan";
			break;
		case T_NestPath:
			ptype = "NestLoop";
			join = true;
			break;
		case T_MergePath:
			ptype = "MergeJoin";
			join = true;
			break;
		case T_HashPath:
			ptype = "HashJoin";
			join = true;
			break;
		case T_AppendPath:
			ptype = "Append";
			break;
		case T_MergeAppendPath:
			ptype = "MergeAppend";
			break;
		case T_GroupResultPath:
			ptype = "GroupResult";
			break;
		case T_MaterialPath:
			ptype = "Material";
			subpath = ((MaterialPath *) path)->subpath;
			break;
		case T_MemoizePath:
			ptype = "Memoize";
			subpath = ((MemoizePath *) path)->subpath;
			break;
		case T_UniquePath:
			ptype = "Unique";
			subpath = ((UniquePath *) path)->subpath;
			break;
		case T_GatherPath:
			ptype = "Gather";
			subpath = ((GatherPath *) path)->subpath;
			break;
		case T_GatherMergePath:
			ptype = "GatherMerge";
			subpath = ((GatherMergePath *) path)->subpath;
			break;
		case T_ProjectionPath:
			ptype = "Projection";
			subpath = ((ProjectionPath *) path)->subpath;
			break;
		case T_ProjectSetPath:
			ptype = "ProjectSet";
			subpath = ((ProjectSetPath *) path)->subpath;
			break;
		case T_SortPath:
			ptype = "Sort";
			subpath = ((SortPath *) path)->subpath;
			break;
		case T_IncrementalSortPath:
			ptype = "IncrementalSort";
			subpath = ((SortPath *) path)->subpath;
			break;
		case T_GroupPath:
			ptype = "Group";
			subpath = ((GroupPath *) path)->subpath;
			break;
		case T_UpperUniquePath:
			ptype = "UpperUnique";
			subpath = ((UpperUniquePath *) path)->subpath;
			break;
		case T_AggPath:
			ptype = "Agg";
			subpath = ((AggPath *) path)->subpath;
			break;
		case T_GroupingSetsPath:
			ptype = "GroupingSets";
			subpath = ((GroupingSetsPath *) path)->subpath;
			break;
		case T_MinMaxAggPath:
			ptype = "MinMaxAgg";
			break;
		case T_WindowAggPath:
			ptype = "WindowAgg";
			subpath = ((WindowAggPath *) path)->subpath;
			break;
		case T_SetOpPath:
			ptype = "SetOp";
			subpath = ((SetOpPath *) path)->subpath;
			break;
		case T_RecursiveUnionPath:
			ptype = "RecursiveUnion";
			break;
		case T_LockRowsPath:
			ptype = "LockRows";
			subpath = ((LockRowsPath *) path)->subpath;
			break;
		case T_ModifyTablePath:
			ptype = "ModifyTable";
			break;
		case T_LimitPath:
			ptype = "Limit";
			subpath = ((LimitPath *) path)->subpath;
			break;
		default:
			ptype = "???Path";
			break;
	}

	custom_snprintf(stream, buf_size, "{\"Node Type\": \"%s\",", ptype);
	custom_snprintf(stream, buf_size, "\"Node Type ID\": \"%d\",", path->type);
	if (path->parent)
	{
		custom_snprintf(stream, buf_size, "\"Relation IDs\": \"");
		debug_print_relids(root, path->parent->relids, stream, buf_size);
		custom_snprintf(stream, buf_size, "\",");

		// Get context
		List *context = NIL;

		if (path->parent->baserestrictinfo)
		{	
			context = create_context(root);
			custom_snprintf(stream, buf_size, "\"Base Restrict Info\": \"");
			debug_print_restrictclauses(root, path->parent->baserestrictinfo, context, stream, buf_size);
			custom_snprintf(stream, buf_size, "\",");
		}

		if (path->parent->joininfo)
		{	
			if (!context)
				context = create_context(root);
			custom_snprintf(stream, buf_size, "\"Join Info\": \"");
			debug_print_restrictclauses(root, path->parent->joininfo, context, stream, buf_size);
			custom_snprintf(stream, buf_size, "\",");
		}
		if (context)
			delete_context(context);
	}
	if (path->param_info)
	{
    	custom_snprintf(stream, buf_size, "\"Required Outer\": \"");
		debug_print_relids(root, path->param_info->ppi_req_outer, stream, buf_size);
		custom_snprintf(stream, buf_size, "\",");
	}
	if (path->parent->reloptkind == RELOPT_JOINREL)
	{	
		custom_snprintf(stream, buf_size, "\"Join Cond\": \"");
		debug_print_joincond(root, path->parent, stream, buf_size);
		custom_snprintf(stream, buf_size, "\",");
	}
	if (path->pathtarget)
	{	
		custom_snprintf(stream, buf_size, "\"Path Target\": \"");
		PathTarget *pathtarget = path->pathtarget;
        ListCell *lc_expr;
		bool first = true;
        foreach(lc_expr, pathtarget->exprs) {
			if (!first)
				custom_snprintf(stream, buf_size, ", ");
            Node *expr = (Node *) lfirst(lc_expr);
            debug_print_expr(expr, root->parse->rtable, stream, buf_size);
			first = false;
        }
		custom_snprintf(stream, buf_size, "\",");
	}

  custom_snprintf(stream, buf_size, "\"Startup Cost\": %f,", path->startup_cost);
  custom_snprintf(stream, buf_size, "\"Total Cost\": %f,", path->total_cost);
  custom_snprintf(stream, buf_size, "\"Plan Rows\": %f,", path->rows);
  custom_snprintf(stream, buf_size, "\"Plan Width\": %d", path->pathtarget->width);


	// if (path->pathkeys)
	// {
	// 	custom_snprintf(stream, buf_size, "\"Pathkeys\": ");
	// 	print_pathkeys(path->pathkeys, root->parse->rtable);
	// }

	if (join)
	{
		JoinPath *jp = (JoinPath *)path;

		// for (i = 0; i < indent; i++)
		// 	appendStringInfoString(&buf, "\t");
		// appendStringInfoString(&buf, "  clauses: ");
		// print_restrictclauses(root, jp->joinrestrictinfo);
		// appendStringInfoString(&buf, "\n");

		if (IsA(path, MergePath))
		{
			MergePath *mp = (MergePath *)path;

			custom_snprintf(stream, buf_size, ", \"Sort Outer\": %d,", ((mp->outersortkeys) ? 1 : 0));
			custom_snprintf(stream, buf_size, "\"Sort Inner\": %d,", ((mp->innersortkeys) ? 1 : 0));
			custom_snprintf(stream, buf_size, "\"Materialize Inner\": %d", ((mp->materialize_inner) ? 1 : 0));
		}

    custom_snprintf(stream, buf_size, ", \"Plans\": [");
		debug_print_path(root, jp->outerjoinpath, indent + 1, stream, buf_size);
    custom_snprintf(stream, buf_size, ", ");
		debug_print_path(root, jp->innerjoinpath, indent + 1, stream, buf_size);
    custom_snprintf(stream, buf_size, "]");
	}

	if (subpath)
	{ 
	// Plans or SubPlan
    custom_snprintf(stream, buf_size, ", \"Plans\": [");
		debug_print_path(root, subpath, indent + 1, stream, buf_size);
		custom_snprintf(stream, buf_size, "]");
	}
	
  custom_snprintf(stream, buf_size, "}");
}

static char* plan_to_json(LeonState * state, PlannerInfo *root, Path* plan, char* leon_query_name) {

	MemoryContext oldContext = MemoryContextSwitchTo(state->leonContext);
    size_t buf_size = 1024;
    char *buf = (char *) palloc0(buf_size);

    custom_snprintf(&buf, &buf_size, "{\"QueryId\": \"%s\", \"Plan\": ", leon_query_name);
    debug_print_path(root, plan, 0, &buf, &buf_size); 
    custom_snprintf(&buf, &buf_size, "}\n");

    MemoryContextSwitchTo(oldContext);
    return buf; // buf will be freed by the caller
}


static bool should_leon_optimize(LeonState * state, int level, int levels_needed, PlannerInfo * root, RelOptInfo * rel, char* leon_query_name) {

	bool should_optimize = false;
	MemoryContext oldContext = MemoryContextSwitchTo(state->leonContext);

	int conn_fd = connect_to_leon(state->leon_host, state->leon_port);
	if (conn_fd < 0) {
		elog(WARNING, "Unable to connect to LEON server %d, should optimize will be cancelled", state->leon_port);
		exit(0);
	}
	write_all_to_socket(conn_fd, START_SHOULD_OPT_MESSAGE);

	size_t buf_size = 1024;
	char *buf = (char *) palloc0(buf_size);

	custom_snprintf(&buf, &buf_size, "{\"QueryId\": \"%s\", \"Relation IDs\": \"", leon_query_name);
	debug_print_relids(root, rel->relids, &buf, &buf_size);
	custom_snprintf(&buf, &buf_size, "\",");
	custom_snprintf(&buf, &buf_size, "\"Current Level\": %d,", level);
	custom_snprintf(&buf, &buf_size, "\"Levels Needed\": %d", levels_needed);
	custom_snprintf(&buf, &buf_size, "}\n");

	write_all_to_socket(conn_fd, buf);
	pfree(buf);
	
	write_all_to_socket(conn_fd, TERMINAL_MESSAGE);
	shutdown(conn_fd, SHUT_WR);

	char *response = (char *)palloc0(5 * sizeof(char));

	if (read(conn_fd, response, 5) > 0) 
	{
		if (strcmp(response, "1") == 0)
		{	
			should_optimize = true;
			goto cleanup;
		}
		else if (strcmp(response, "0") == 0)
		{	
			should_optimize = false;
			goto cleanup;
		}
		else
		{	// Show response
			elog(WARNING, "LEON could not read the response: %s from the server.", response);
			should_optimize = false;
			goto cleanup;
		}
	}
	else
	{
		elog(WARNING, "LEON could not read the response from the server.");
		should_optimize = false;
		goto cleanup;
	}

cleanup:

	pfree(response);
	shutdown(conn_fd, SHUT_RDWR);
	if (conn_fd > 0) close(conn_fd);
	MemoryContextSwitchTo(oldContext);
	return should_optimize;
}

static int compare_paths(const void *a, const void *b)
{
    Path *path_a = *(Path **)a;
    Path *path_b = *(Path **)b;

    // 根据路径的 cost 进行比较
    if (path_a->total_cost < path_b->total_cost)
        return -1;
    else if (path_a->total_cost > path_b->total_cost)
        return 1;
    else
        return 0;
}

// 对 savedpaths 列表按照 cost 进行排序
void sort_savedpaths_by_cost(RelOptInfo *joinrel)
{
    if (joinrel->savedpaths != NIL)
    {
        list_sort(joinrel->savedpaths, compare_paths);
    }
}

void keep_first_50_rels(RelOptInfo *joinrel, int free_size)
{
    if (joinrel->savedpaths != NIL)
    {
		if(list_length(joinrel->savedpaths) > free_size)
		{
			// 使用 list_truncate 保留前 50 个元素
			// Here we only keep the first 50 paths
			// The rest of the paths will not be freed
        	joinrel->savedpaths = list_truncate(joinrel->savedpaths, free_size);
		}
        
    }
}

#endif