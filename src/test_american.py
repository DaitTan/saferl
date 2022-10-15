from lnn import Predicate, Variable, Join, And, Exists, Implies, ForAll, Model, Fact, Predicates, Equivalent, World, Loss


def test_1():
    """The 'American' theorem proving example with outer joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2)
    missile = Predicate("missile")
    american = Predicate("american")
    enemy = Predicate("enemy", arity=2)
    hostile = Predicate("hostile")
    criminal = Predicate("criminal")
    weapon = Predicate("weapon")
    sells = Predicate("sells", arity=3)

    # Define and add the background knowledge to  the model.

    query = Exists(x, criminal(x), join=Join.OUTER)

    model.add_knowledge(
        ForAll(
            x,
            y,
            Implies(enemy(x, y, bind={y: "America"}), hostile(x), join=Join.OUTER),
            join=Join.OUTER,
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(
                    american(x), weapon(y), sells(x, y, z), hostile(z), join=Join.OUTER
                ),
                criminal(x),
                join=Join.OUTER,
            ),
            join=Join.OUTER,
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(missile(x), owns(y, x, bind={y: "Nono"}), join=Join.OUTER),
                sells(z, x, y, bind={z: "West", y: "Nono"}),
                join=Join.OUTER,
            ),
            join=Join.OUTER,
        ),
        ForAll(x, Implies(missile(x), weapon(x), join=Join.OUTER), join=Join.OUTER),
    )

    model.set_query(query)

    # Add facts to model.
    model.add_data(
        {
            owns: {("Nono", "M1"): Fact.TRUE},
            missile: {"M1": Fact.TRUE},
            american: {"West": Fact.TRUE},
            enemy: {("Nono", "America"): Fact.TRUE},
        }
    )

    model.infer()
    model.print()
    GT_o = dict([("West", Fact.TRUE)])
    model.print()
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"


def test_2():
    """The 'American' theorem proving example with  inner joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2)
    missile = Predicate("missile")
    american = Predicate("american")
    enemy = Predicate("enemy", arity=2)
    hostile = Predicate("hostile")
    criminal = Predicate("criminal")
    weapon = Predicate("weapon")
    sells = Predicate("sells", arity=3)

    # Define and add the background knowledge to  the model.

    query = Exists(x, criminal(x))

    model.add_knowledge(
        ForAll(
            x,
            y,
            Implies(enemy(x, y, bind={y: "America"}), hostile(x)),
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(american(x), weapon(y), sells(x, y, z), hostile(z)),
                criminal(x),
            ),
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(missile(x), owns(y, x, bind={y: "Nono"})),
                sells(z, x, y, bind={z: "West", y: "Nono"}),
            ),
        ),
        ForAll(x, Implies(missile(x), weapon(x))),
    )

    model.set_query(query)

    # Add facts to model.
    model.add_data(
        {
            owns: {("Nono", "M1"): Fact.TRUE},
            missile: {"M1": Fact.TRUE},
            american: {"West": Fact.TRUE},
            enemy: {("Nono", "America"): Fact.TRUE},
        }
    )

    model.infer()
    model.print()
    GT_o = dict([("West", Fact.TRUE)])
    model.print()
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"
    print(f'I am ground {model[query].groundings}')

def smokers():
    x, y = map(Variable, ('x', 'y'))

    model = Model()

    Smokes = Predicate('Smokes')
    Friends = Predicate('Friends', arity=2)
    Cancer = Predicate('Cancer', world=World.CLOSED)

    Smoking_causes_Cancer = Implies(Smokes(x), Cancer(x))
    Smokers_befriend_Smokers = Implies(Friends(x, y), Equivalent(Smokes(x), Smokes(y)))

    formulae = [
        Smoking_causes_Cancer,
        Smokers_befriend_Smokers
    ]
    model.add_knowledge(*formulae, world=World.AXIOM)

    model.add_data({
        Friends: {
            ('Anna', 'Bob'): Fact.TRUE,
            ('Bob', 'Anna'): Fact.TRUE,
            ('Anna', 'Edward'): Fact.TRUE,
            ('Edward', 'Anna'): Fact.TRUE,
            ('Anna', 'Frank'): Fact.TRUE,
            ('Frank', 'Anna'): Fact.TRUE,
            ('Bob', 'Chris'): Fact.TRUE},
        Smokes: {
            'Anna': Fact.TRUE,
            'Edward': Fact.TRUE,
            'Frank': Fact.TRUE,
            'Gary': Fact.TRUE},
        Cancer: {
            'Anna': Fact.TRUE,
            'Edward': Fact.TRUE}
        })

    # model.infer()
    # model.print()
    
    model.add_labels({
    Smokes: {
        'Bob': Fact.TRUE,
        'Nick': Fact.TRUE}
    })

    # train the model and output results
    # model.infer()
    model.train(losses=[Loss.SUPERVISED, Loss.CONTRADICTION])
    model.print(params=True)
    print(Cancer.state("Bob"))
    print(Smokes.state("Bob"))
    print(Cancer.state("Ivan"))
    print(Smokes.state("Ivan"))


if __name__ == "__main__":
    #test_1()
    # test_2()
    # print("success")
    smokers()
