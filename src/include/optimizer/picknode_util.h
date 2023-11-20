#ifndef PICKNODE_UTIL_H
#define PICKNODE_UTIL_H
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>


typedef struct PickNodeState
{
	int node_level;
	Relids node_relids;
} PickNodeState;

// static PickNodeState *current_picknode_state = NULL;
extern PickNodeState *current_picknode_state;

/*
 * Return index of relation which matches given aliasname, or 0 if not found.
 * If same aliasname was used multiple times in a query, return -1.
 */

static int
RelnameCmp(const void *a, const void *b)
{
	const char *relnamea = *((const char **) a);
	const char *relnameb = *((const char **) b);

	return strcmp(relnamea, relnameb);
}

static int
find_relid_aliasname(PlannerInfo *root, char *aliasname, List *initial_rels)
{
	int		i;
	Index	found = 0;

	for (i = 1; i < root->simple_rel_array_size; i++)
	{
		ListCell   *l;

		if (root->simple_rel_array[i] == NULL)
			continue;

		Assert(i == root->simple_rel_array[i]->relid);

		if (RelnameCmp(&aliasname,
					   &root->simple_rte_array[i]->eref->aliasname) != 0)
			continue;

		foreach(l, initial_rels)
		{
			RelOptInfo *rel = (RelOptInfo *) lfirst(l);

			if (rel->reloptkind == RELOPT_BASEREL)
			{
				if (rel->relid != i)
					continue;
			}
			else
			{
				Assert(rel->reloptkind == RELOPT_JOINREL);

				if (!bms_is_member(i, rel->relids))
					continue;
			}

			if (found != 0)
			{	
				elog(WARNING, "Relation name \"%s\" is ambiguous.", aliasname);
				return -1;
			}

			found = i;
			break;
		}

	}

	return found;
}

static Relids
create_bms_of_relids(PlannerInfo *root, List *initial_rels,
		int nrels, char **relnames)
{
	int		relid;
	Relids	relids = NULL;
	int		j;
	char   *relname;

	for (j = 0; j < nrels; j++)
	{
		relname = relnames[j];

		relid = find_relid_aliasname(root, relname, initial_rels);

		if (relid == -1)
			elog(WARNING, "Relation name \"%s\" is ambiguous.", relname);

		/*
		 * the aliasname is not found(relid == 0) or same aliasname was used
		 * multiple times in a query(relid == -1)
		 */
		if (relid <= 0)
		{
			relids = NULL;
			break;
		}
		if (bms_is_member(relid, relids))
		{	
			elog(WARNING, "Relation name \"%s\" is duplicated.", relname);
			break;
		}

		relids = bms_add_member(relids, relid);
	}
	return relids;
}

#endif